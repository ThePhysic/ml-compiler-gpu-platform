"""Original graph passes, deterministic scheduling and interval arena allocation.

All rewrites operate on a private graph copy.  Scheduling and allocation are
backend-independent; tile configurations specialize the Triton lowering.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable

from .errors import CompileError, ResourceError
from .ir import Graph, Node
from .reference import evaluate_node


ALIGNMENT = 256
MAX_FUSION_STEPS = 16
MAX_FOLDED_BYTES = 64 * 1024 * 1024
ELEMENTWISE = frozenset({"add", "mul", "relu", "gelu"})


def _align(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(frozen=True)
class Allocation:
    offset: int
    size: int
    start: int
    end: int

    @property
    def reserved_size(self) -> int:
        return _align(self.size)

    def to_dict(self) -> dict:
        return {**asdict(self), "reserved_size": self.reserved_size}


@dataclass
class MemoryPlan:
    allocations: dict[str, Allocation]
    peak_bytes: int
    naive_bytes: int
    external_values: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "alignment": ALIGNMENT,
            "peak_bytes": self.peak_bytes,
            "naive_bytes": self.naive_bytes,
            "external_values": list(self.external_values),
            "allocations": {name: alloc.to_dict() for name, alloc in self.allocations.items()},
        }


@dataclass
class ExecutionPlan:
    graph: Graph
    nodes: list[Node]
    memory: MemoryPlan
    passes: list[dict]
    tiles: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "graph": self.graph.to_dict(),
            "schedule": [
                {"op": n.op, "inputs": list(n.inputs), "output": n.output, "attrs": deepcopy(n.attrs)}
                for n in self.nodes
            ],
            "memory": self.memory.to_dict(),
            "passes": deepcopy(self.passes),
            "tiles": deepcopy(self.tiles),
        }

    def format(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _users(nodes: Iterable[Node]) -> dict[str, list[Node]]:
    users = defaultdict(list)
    for node in nodes:
        for value in node.inputs:
            users[value].append(node)
    return users


def _prune(graph: Graph) -> Graph:
    """Remove unobservable computations; preserve the declared input signature."""
    needed = set(graph.outputs)
    kept = []
    for node in reversed(graph.ordered_nodes()):
        if node.output in needed:
            kept.append(node)
            needed.update(node.inputs)
    graph.nodes = list(reversed(kept))
    graph.constants = {name: value for name, value in graph.constants.items() if name in needed}
    return graph


def fold_constants(graph: Graph) -> Graph:
    """Fold bounded constant-only subgraphs using precisely the reference semantics."""
    specs = graph.specs()
    nodes = []
    for node in graph.ordered_nodes():
        if all(name in graph.constants for name in node.inputs) and specs[node.output].nbytes <= MAX_FOLDED_BYTES:
            try:
                graph.constants[node.output] = evaluate_node(node, graph.constants)
            except Exception as exc:
                raise CompileError(f"constant folding failed for {node.output}: {exc}") from exc
        else:
            nodes.append(node)
    graph.nodes = nodes
    return _prune(graph)


def _substitute_inputs(node: Node, aliases: dict[str, str]) -> None:
    """Substitute graph operands without capturing nested region-local names."""
    old_inputs = node.inputs
    new_inputs = tuple(aliases.get(name, name) for name in old_inputs)
    node.inputs = new_inputs
    if node.op != "fused_elementwise" or new_inputs == old_inputs:
        return
    steps = node.attrs["steps"]
    # Locals may have the same spelling as a graph value outside this region.
    # Renaming an external a -> x must not capture a local definition named x.
    bindings = dict(zip(old_inputs, new_inputs))
    external = set(new_inputs)
    used = external | set(old_inputs) | {step["output"] for step in steps}
    for index, step in enumerate(steps):
        step["inputs"] = [bindings[name] for name in step["inputs"]]
        original = step["output"]
        renamed = original
        if original in external:
            suffix = 0
            renamed = f"__region_{node.output}_{index}_{suffix}"
            while renamed in used:
                suffix += 1
                renamed = f"__region_{node.output}_{index}_{suffix}"
            used.add(renamed)
        bindings[original] = renamed
        step["output"] = renamed


def simplify(graph: Graph) -> Graph:
    """Bypass internal identity/copy operations, preserving public output names."""
    specs = graph.specs()
    aliases: dict[str, str] = {}
    nodes = []
    for node in graph.ordered_nodes():
        _substitute_inputs(node, aliases)
        trivial = node.op == "identity"
        if node.op == "reshape":
            trivial = specs[node.inputs[0]].shape == specs[node.output].shape
        elif node.op == "transpose":
            axes = tuple(node.attrs.get("axes", reversed(range(len(specs[node.inputs[0]].shape)))))
            trivial = axes == tuple(range(len(axes)))
        if trivial and node.output not in graph.outputs:
            aliases[node.output] = node.inputs[0]
        else:
            nodes.append(node)
    graph.nodes = nodes
    return _prune(graph)


def fuse_matmul(graph: Graph) -> Graph:
    """Fuse single-use matmul → optional row bias → optional activation regions."""
    specs = graph.specs()
    ordered = graph.ordered_nodes()
    users = _users(ordered)
    outputs = set(graph.outputs)
    consumed: set[str] = set()
    replacements: dict[str, Node] = {}
    for node in ordered:
        if node.op != "matmul" or node.output in consumed:
            continue
        chain = [node]
        bias = None
        activation = "none"
        current = node
        if current.output not in outputs and len(users[current.output]) == 1:
            nxt = users[current.output][0]
            if nxt.op == "add" and nxt.output not in consumed:
                candidate = nxt.inputs[1] if nxt.inputs[0] == current.output else nxt.inputs[0]
                n = specs[current.output].shape[-1]
                if candidate != current.output and specs[candidate].shape in {(n,), (1, n)}:
                    chain.append(nxt)
                    bias = candidate
                    current = nxt
        if current.output not in outputs and len(users[current.output]) == 1:
            nxt = users[current.output][0]
            if nxt.op in {"relu", "gelu"} and nxt.output not in consumed:
                chain.append(nxt)
                activation = nxt.op
        if len(chain) < 2:
            continue
        fused = Node(
            "fused_matmul", tuple(node.inputs) + ((bias,) if bias is not None else ()),
            chain[-1].output, {"activation": activation},
        )
        consumed.update(n.output for n in chain)
        replacements[chain[-1].output] = fused
    graph.nodes = [
        replacements[node.output] if node.output in replacements else node
        for node in ordered if node.output not in consumed or node.output in replacements
    ]
    return graph


def fuse_elementwise(graph: Graph) -> Graph:
    """Fuse linear same-shape SSA regions; external operands may broadcast."""
    ordered = graph.ordered_nodes()
    specs = graph.specs()
    users = _users(ordered)
    outputs = set(graph.outputs)
    consumed: set[str] = set()
    replacements: dict[str, Node] = {}
    for node in ordered:
        if node.op not in ELEMENTWISE or node.output in consumed:
            continue
        chain = [node]
        while len(chain) < MAX_FUSION_STEPS:
            current = chain[-1]
            if current.output in outputs or len(users[current.output]) != 1:
                break
            nxt = users[current.output][0]
            if nxt.op not in ELEMENTWISE or nxt.output in consumed:
                break
            if specs[nxt.output] != specs[node.output]:
                break
            chain.append(nxt)
        if len(chain) < 2:
            continue
        internal = {n.output for n in chain}
        external = tuple(dict.fromkeys(name for n in chain for name in n.inputs if name not in internal))
        fused = Node("fused_elementwise", external, chain[-1].output, {"steps": [
            {"op": n.op, "inputs": list(n.inputs), "output": n.output, "attrs": deepcopy(n.attrs)}
            for n in chain
        ]})
        consumed.update(internal)
        replacements[chain[-1].output] = fused
    graph.nodes = [
        replacements[node.output] if node.output in replacements else node
        for node in ordered if node.output not in consumed or node.output in replacements
    ]
    return graph


def schedule_nodes(graph: Graph) -> list[Node]:
    """Deterministic greedy schedule minimizing predicted resident-buffer pressure.

    The score considers the allocation *before* releasing last-use inputs, since
    a GPU operation reads its inputs while writing its distinct output.
    """
    ordered = graph.ordered_nodes()
    specs = graph.specs()
    index = {n.output: i for i, n in enumerate(ordered)}
    pending = list(ordered)
    remaining = Counter(name for n in ordered for name in n.inputs)
    available = set(graph.inputs) | set(graph.constants)
    external = set(available)
    pinned = set(graph.outputs)
    live: set[str] = set()
    resident = 0
    peak = 0
    result = []
    while pending:
        ready = [n for n in pending if all(name in available for name in n.inputs)]
        if not ready:
            raise CompileError("scheduler found a dependency cycle or missing value")

        def score(node: Node) -> tuple[int, int, int]:
            required = _align(specs[node.output].nbytes)
            uses = Counter(node.inputs)
            freed = sum(_align(specs[name].nbytes) for name, count in uses.items()
                        if name in live and name not in pinned and remaining[name] == count)
            retained_output = required if remaining[node.output] or node.output in pinned else 0
            return max(peak, resident + required), resident - freed + retained_output, index[node.output]

        node = min(ready, key=score)
        required = _align(specs[node.output].nbytes)
        peak = max(peak, resident + required)
        resident += required
        live.add(node.output)
        for name in node.inputs:
            remaining[name] -= 1
            if name in live and name not in pinned and remaining[name] == 0:
                live.remove(name)
                resident -= _align(specs[name].nbytes)
        if remaining[node.output] == 0 and node.output not in pinned:
            live.remove(node.output)
            resident -= required
        available.add(node.output)
        pending.remove(node)
        result.append(node)
    return result


def _coalesce(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for offset, size in sorted(blocks):
        if merged and merged[-1][0] + merged[-1][1] == offset:
            prior, amount = merged[-1]
            merged[-1] = (prior, amount + size)
        else:
            merged.append((offset, size))
    return merged


def plan_memory(graph: Graph, nodes: list[Node] | None = None) -> MemoryPlan:
    """Allocate aligned intervals by best fit, splitting/coalescing free ranges."""
    if nodes is None:
        nodes = graph.ordered_nodes()
    specs = graph.specs()
    positions = {node.output: i for i, node in enumerate(nodes)}
    ends = dict(positions)
    for i, node in enumerate(nodes):
        for name in node.inputs:
            if name in ends:
                ends[name] = max(ends[name], i)
    for name in graph.outputs:
        if name in ends:
            ends[name] = len(nodes)
    active: dict[str, Allocation] = {}
    allocations: dict[str, Allocation] = {}
    free: list[tuple[int, int]] = []
    arena = 0
    naive = 0
    for i, node in enumerate(nodes):
        for name, allocation in list(active.items()):
            if allocation.end < i:
                free.append((allocation.offset, allocation.reserved_size))
                del active[name]
        free = _coalesce(free)
        size = specs[node.output].nbytes
        required = _align(size)
        naive += required
        candidates = [(amount, offset, slot) for slot, (offset, amount) in enumerate(free) if amount >= required]
        if candidates:
            amount, offset, slot = min(candidates)
            del free[slot]
            if amount > required:
                free.append((offset + required, amount - required))
        else:
            offset = arena
            arena += required
        allocation = Allocation(offset, size, i, ends[node.output])
        allocations[node.output] = allocation
        active[node.output] = allocation
    return MemoryPlan(allocations, arena, naive, tuple(graph.inputs) + tuple(graph.constants))


def choose_tiles(graph: Graph) -> dict[str, dict]:
    """Select from explicit tile candidates using a documented static cost proxy.

    This is a shape heuristic, not a hardware timing prediction or autotuner.
    The benchmark tool can compare configurations against these defaults.
    """
    specs = graph.specs()
    configs = {}
    candidates = [(16, 32, 32, 4), (32, 64, 32, 4), (64, 64, 32, 4), (32, 128, 32, 4), (64, 128, 32, 8)]
    for node in graph.nodes:
        if node.op not in {"matmul", "fused_matmul"}:
            continue
        m, k = specs[node.inputs[0]].shape
        _, n = specs[node.inputs[1]].shape

        def cost(candidate: tuple[int, int, int, int]) -> tuple[float, tuple[int, int, int, int]]:
            bm, bn, bk, warps = candidate
            programs = math.ceil(m / bm) * math.ceil(n / bn)
            padded_flops = programs * bm * bn * math.ceil(k / bk) * bk * 2
            accum_per_thread = bm * bn / (warps * 32)
            excess_registers = max(0, accum_per_thread - 48)
            return padded_flops + programs * (256 * k + 512) + excess_registers * programs * 4096, candidate

        bm, bn, bk, warps = min(candidates, key=cost)
        configs[node.output] = {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": warps}
    return configs


def compile_graph(graph: Graph, optimize: bool = True, memory_budget: int | None = None) -> ExecutionPlan:
    """Validate, optimize, schedule, specialize and allocate a private graph copy."""
    if not isinstance(optimize, bool):
        raise CompileError("optimize must be a bool")
    if memory_budget is not None and (isinstance(memory_budget, bool) or not isinstance(memory_budget, int) or memory_budget < 0):
        raise CompileError("memory_budget must be a nonnegative integer number of bytes")
    work = deepcopy(graph)
    work.validate()
    passes = []
    if optimize:
        for name, transform in [
            ("constant_folding", fold_constants),
            ("simplification", simplify),
            ("matmul_epilogue_fusion", fuse_matmul),
            ("elementwise_fusion", fuse_elementwise),
            ("dead_code_elimination", _prune),
        ]:
            before = work.to_dict()
            count = len(work.nodes)
            work = transform(work)
            work.validate()
            after = work.to_dict()
            passes.append({"pass": name, "changed": before != after,
                           "before_nodes": count, "after_nodes": len(work.nodes),
                           "before": before, "after": after})
    nodes = schedule_nodes(work) if optimize else work.ordered_nodes()
    memory = plan_memory(work, nodes)
    if memory_budget is not None and memory.peak_bytes > memory_budget:
        raise ResourceError(f"temporary arena requires {memory.peak_bytes} bytes, budget is {memory_budget} bytes")
    tiles = choose_tiles(work)
    passes.append({"pass": "dependency_scheduling", "changed": [n.output for n in nodes] != [n.output for n in work.ordered_nodes()],
                   "order": [n.output for n in nodes]})
    passes.append({"pass": "memory_planning", "changed": memory.peak_bytes < memory.naive_bytes,
                   "arena_bytes": memory.peak_bytes, "naive_bytes": memory.naive_bytes})
    return ExecutionPlan(work, nodes, memory, passes, tiles)
