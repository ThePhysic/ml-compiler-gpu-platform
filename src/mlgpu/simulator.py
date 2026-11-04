"""A tiled tensor accelerator model, deliberately independent of physical GPUs.

The functional interpreter executes emitted DMA, matrix, vector and store
instructions. Cycles belong to a separately specified, synthetic resource model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from typing import Any, Mapping

import numpy as np

from .errors import ExecutionError, ResourceError
from .ir import Graph, Node


@dataclass(frozen=True)
class HardwareConfig:
    local_memory_bytes: int = 65536
    banks: int = 8
    compute_units: int = 2
    transfer_units: int = 1
    flops_per_cycle_per_unit: int = 256
    bytes_per_cycle_per_transfer: int = 32
    instruction_setup_cycles: int = 4
    core_clock_ghz: float = 1.0
    max_tile: int = 32

    def __post_init__(self):
        for name in ("local_memory_bytes", "banks", "compute_units", "transfer_units",
                     "flops_per_cycle_per_unit", "bytes_per_cycle_per_transfer",
                     "instruction_setup_cycles", "max_tile"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ResourceError(f"simulator hardware {name} must be a positive integer")
        if self.banks < 4 or self.local_memory_bytes < self.banks * 4:
            raise ResourceError("simulator requires >=4 SRAM banks with >=4 bytes per bank")
        if isinstance(self.core_clock_ghz, bool) or not isinstance(self.core_clock_ghz, (int, float)) or not math.isfinite(self.core_clock_ghz) or self.core_clock_ghz <= 0:
            raise ResourceError("simulator core_clock_ghz must be positive and finite")

    @property
    def bank_bytes(self) -> int:
        return self.local_memory_bytes // self.banks

    @property
    def contexts(self) -> int:
        return min(self.compute_units, self.banks // 4)

    def to_dict(self):
        return {**asdict(self), "bank_bytes": self.bank_bytes,
                "tile_contexts": self.contexts,
                "unusable_remainder_bytes": self.local_memory_bytes % self.banks}


@dataclass(frozen=True)
class Instruction:
    id: int
    op: str
    deps: tuple[int, ...]
    start_cycle: int
    end_cycle: int
    resource: str
    banks: tuple[int, ...]
    transfer_bytes: int = 0
    flops: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class SimulatorPlan:
    graph: Graph
    hardware: HardwareConfig
    instructions: tuple[Instruction, ...]
    optimized: bool
    tiles: dict[str, dict[str, int]] = field(default_factory=dict)
    scheduled_nodes: tuple[Node, ...] = ()

    def to_dict(self):
        return {"backend": "synthetic_tensor_accelerator", "optimized": self.optimized,
                "hardware": self.hardware.to_dict(), "graph": self.graph.to_dict(),
                "tiles": self.tiles, "node_order": [n.output for n in self.scheduled_nodes],
                "instructions": [i.to_dict() for i in self.instructions],
                "estimate": estimate(self)}

    def format(self):
        return json.dumps(self.to_dict(), indent=2)


class _Emitter:
    def __init__(self, hw: HardwareConfig):
        self.hw = hw
        self.instructions: list[Instruction] = []
        self.compute = [0] * hw.compute_units
        self.transfer = [0] * hw.transfer_units
        self.bank_ready = [0] * hw.banks
        self.bank_previous: dict[int,int] = {}

    def add(self, op, deps=(), banks=(), transfer_bytes=0, flops=0, **attrs):
        banks = tuple(sorted(set(banks)))
        deps = tuple(sorted(set(deps) | {self.bank_previous[b] for b in banks if b in self.bank_previous}))
        transfer = op in {"load", "load_group", "store", "store_group"}
        pool, prefix = (self.transfer, "dma") if transfer else (self.compute, "compute")
        dependency_ready = max((self.instructions[d].end_cycle for d in deps), default=0)
        ready = max(dependency_ready, max((self.bank_ready[b] for b in banks), default=0))
        unit = min(range(len(pool)), key=lambda u: (max(ready, pool[u]), u))
        start = max(ready, pool[unit])
        work = transfer_bytes if transfer else flops
        rate = self.hw.bytes_per_cycle_per_transfer if transfer else self.hw.flops_per_cycle_per_unit
        end = start + self.hw.instruction_setup_cycles + math.ceil(work / rate)
        pool[unit] = end
        for bank in banks:
            self.bank_ready[bank] = end
        instruction = Instruction(len(self.instructions), op, deps, start, end,
                                  f"{prefix}:{unit}", banks, transfer_bytes, flops, attrs)
        self.instructions.append(instruction)
        for bank in banks:
            self.bank_previous[bank] = instruction.id
        return instruction.id


def _reschedule(instructions,hw):
    """Global earliest-ready list scheduling fills independent DMA/compute gaps."""
    pending={ins.id:set(ins.deps) for ins in instructions}
    children={ins.id:[] for ins in instructions}
    for ins in instructions:
        for dep in ins.deps: children[dep].append(ins.id)
    ready={i for i,deps in pending.items() if not deps}
    result={}
    pools={"dma":[0]*hw.transfer_units,"compute":[0]*hw.compute_units}
    bank_ready=[0]*hw.banks
    while ready:
        choices=[]
        for index in ready:
            ins=instructions[index]
            prefix=ins.resource.split(":")[0]
            dependency_ready=max((result[d].end_cycle for d in ins.deps),default=0)
            available=max(dependency_ready,max((bank_ready[b] for b in ins.banks),default=0))
            unit=min(range(len(pools[prefix])),key=lambda u:(max(available,pools[prefix][u]),u))
            choices.append((max(available,pools[prefix][unit]),index,unit,prefix))
        start,index,unit,prefix=min(choices)
        ready.remove(index)
        ins=instructions[index]
        end=start+ins.end_cycle-ins.start_cycle
        result[index]=replace(ins,start_cycle=start,end_cycle=end,resource=f"{prefix}:{unit}")
        pools[prefix][unit]=end
        for bank in ins.banks: bank_ready[bank]=end
        for child in children[index]:
            pending[child].remove(index)
            if not pending[child]: ready.add(child)
    if len(result)!=len(instructions):
        raise ResourceError("instruction dependency cycle")
    return tuple(result[i] for i in range(len(instructions)))


def _partition_bytes(size, banks, capacity):
    result = {}
    for bank in banks:
        result[str(bank)] = min(size, capacity)
        size -= result[str(bank)]
    if size:
        raise ResourceError("tensor tile exceeds packed SRAM bank capacity")
    return result


def _matmul_tile(shape_a, shape_b, itemsize, hw, optimize):
    m, k = shape_a
    n = shape_b[1]
    limit = hw.max_tile if optimize else min(8, hw.max_tile)
    choices = sorted({1, *[min(limit, t) for t in (2, 4, 8, 16, 32, 64)]}, reverse=True)
    for tile in choices:
        bm, bn, bk = min(m, tile), min(n, tile), min(k, tile)
        if max(bm * bk * itemsize, bk * bn * itemsize, bm * bn * 4) <= hw.bank_bytes:
            return bm, bn, bk
    raise ResourceError("even a one-element matrix tile cannot fit SRAM")


def compile_simulator(graph: Graph, hardware: HardwareConfig | None = None,
                      optimize: bool = True) -> SimulatorPlan:
    """Lower a validated graph to explicit, resource-scheduled instructions."""
    from .compiler import compile_graph

    if not isinstance(optimize, bool):
        raise ResourceError("simulator optimize must be boolean")
    hw = HardwareConfig() if hardware is None else hardware
    if not isinstance(hw, HardwareConfig):
        raise ResourceError("hardware must be HardwareConfig")
    graph.validate()
    compiled = compile_graph(graph, optimize=optimize)
    lowered = compiled.graph
    specs = lowered.specs()
    emitter = _Emitter(hw)
    completed: dict[str, tuple[int, ...]] = {}
    context_done: list[tuple[int, ...]] = [()] * hw.contexts
    tile_counter = 0
    tiles = {}

    for node_index, node in enumerate(compiled.nodes):
        output_spec = specs[node.output]
        input_deps = tuple(d for value in node.inputs for d in completed.get(value, ()))
        stores = []
        if node.op in {"matmul", "fused_matmul"}:
            a_shape, b_shape = (specs[v].shape for v in node.inputs[:2])
            m, k = a_shape
            n = b_shape[1]
            bm, bn, bk = _matmul_tile(a_shape, b_shape, np.dtype(output_spec.dtype).itemsize, hw, optimize)
            tiles[node.output] = {"block_m": bm, "block_n": bn, "block_k": bk}
            for row in range(0, m, bm):
                for col in range(0, n, bn):
                    context = tile_counter % hw.contexts
                    tile_counter += 1
                    a_bank, b_bank, c_bank, bias_bank = range(context * 4, context * 4 + 4)
                    base_deps = (*context_done[context], *input_deps)
                    rows, cols = min(bm, m-row), min(bn, n-col)
                    c_token = f"c:{node.output}:{row}:{col}"
                    c_size = rows * cols * 4
                    last = emitter.add("zero", base_deps, (c_bank,), flops=rows*cols,
                        buffer_writes={str(c_bank): c_token},
                        size_by_bank={str(c_bank): c_size}, shape=[rows, cols], bank=c_bank)
                    for depth in range(0, k, bk):
                        width = min(bk, k-depth)
                        a_token = f"a:{node.output}:{row}:{col}:{depth}"
                        b_token = f"b:{node.output}:{row}:{col}:{depth}"
                        a_size = rows * width * np.dtype(output_spec.dtype).itemsize
                        b_size = width * cols * np.dtype(output_spec.dtype).itemsize
                        load_a = emitter.add("load", (*base_deps, last), (a_bank,), a_size,
                            tensor=node.inputs[0], slices=[[row,row+rows],[depth,depth+width]], bank=a_bank,
                            buffer_writes={str(a_bank): a_token}, size_by_bank={str(a_bank): a_size})
                        load_b = emitter.add("load", (*base_deps, last), (b_bank,), b_size,
                            tensor=node.inputs[1], slices=[[depth,depth+width],[col,col+cols]], bank=b_bank,
                            buffer_writes={str(b_bank): b_token}, size_by_bank={str(b_bank): b_size})
                        last = emitter.add("mma", (last,load_a,load_b), (a_bank,b_bank,c_bank),
                            flops=2*rows*cols*width, a_bank=a_bank,b_bank=b_bank,c_bank=c_bank,
                            buffer_reads={str(a_bank):a_token,str(b_bank):b_token,str(c_bank):c_token},
                            buffer_writes={str(c_bank):c_token}, size_by_bank={str(c_bank):c_size})
                    reads = {str(c_bank):c_token}
                    epi_deps = [last]
                    if len(node.inputs) == 3:
                        bias_token = f"bias:{node.output}:{row}:{col}"
                        bias_size = rows * cols * np.dtype(output_spec.dtype).itemsize
                        load_bias = emitter.add("load", base_deps, (bias_bank,), bias_size,
                            tensor=node.inputs[2], broadcast_shape=[m,n],
                            slices=[[row,row+rows],[col,col+cols]], bank=bias_bank,
                            buffer_writes={str(bias_bank):bias_token},size_by_bank={str(bias_bank):bias_size})
                        epi_deps.append(load_bias)
                        reads[str(bias_bank)] = bias_token
                    last = emitter.add("epilogue", epi_deps, tuple(int(b) for b in reads),
                        flops=rows*cols*(21 if node.attrs.get("activation") == "gelu" else 2),
                        c_bank=c_bank,bias_bank=bias_bank if len(node.inputs)==3 else None,
                        dtype=output_spec.dtype,activation=node.attrs.get("activation","none"),
                        buffer_reads=reads,buffer_writes={str(c_bank):c_token},
                        size_by_bank={str(c_bank):rows*cols*np.dtype(output_spec.dtype).itemsize})
                    store = emitter.add("store", (last,), (c_bank,),
                        rows*cols*np.dtype(output_spec.dtype).itemsize,
                        tensor=node.output,slices=[[row,row+rows],[col,col+cols]],bank=c_bank,
                        buffer_reads={str(c_bank):c_token})
                    stores.append(store)
                    context_done[context] = (store,)
        else:
            descriptor = _generic_descriptor(node, specs, hw, optimize)
            tiles[node.output] = {"output_elements": descriptor["tile_elements"]}
            for start in range(0, descriptor["iteration_size"], descriptor["step"]):
                end = min(start+descriptor["step"], descriptor["iteration_size"])
                context = tile_counter % hw.contexts
                tile_counter += 1
                input_banks = tuple(range(context*4,context*4+3))
                out_bank = context*4+3
                token_in = f"in:{node.output}:{start}"
                token_out = f"out:{node.output}:{start}"
                transfer_size, output_count = _generic_sizes(node, specs, descriptor["kind"], start,end)
                if output_count*np.dtype(output_spec.dtype).itemsize > hw.bank_bytes:
                    raise ResourceError(f"simulator output tile of {node.output} exceeds SRAM bank")
                sizes = _partition_bytes(transfer_size,input_banks,hw.bank_bytes)
                load = emitter.add("load_group", (*input_deps,*context_done[context]), input_banks,
                    transfer_size,node_index=node_index,start=start,end=end,kind=descriptor["kind"],
                    payload_bank=input_banks[0],buffer_writes={str(b):token_in for b in input_banks},
                    size_by_bank=sizes)
                compute = emitter.add("vector", (load,), (*input_banks,out_bank),
                    flops=_generic_flops(node,specs,output_count),node_index=node_index,
                    payload_bank=input_banks[0],output_bank=out_bank,
                    buffer_reads={str(b):token_in for b in input_banks},
                    buffer_writes={str(out_bank):token_out},
                    size_by_bank={str(out_bank):output_count*np.dtype(output_spec.dtype).itemsize})
                store = emitter.add("store_group", (compute,), (out_bank,),
                    output_count*np.dtype(output_spec.dtype).itemsize,tensor=node.output,
                    bank=out_bank,start=start,end=end,kind=descriptor["kind"],
                    buffer_reads={str(out_bank):token_out})
                stores.append(store)
                context_done[context] = (store,)
        completed[node.output] = tuple(stores)
    plan = SimulatorPlan(lowered,hw,_reschedule(emitter.instructions,hw),optimize,tiles,tuple(compiled.nodes))
    _validate_schedule(plan)
    return plan


def _generic_sizes(node, specs, kind, start, end):
    count = end-start
    out = specs[node.output]
    itemsize = np.dtype(out.dtype).itemsize
    if kind == "rows":
        width = out.shape[-1]
        size = count*width*itemsize
        if node.op == "layernorm":
            size += sum(specs[v].nbytes for v in node.inputs[1:])
        return size,count*width
    if kind == "reduction":
        axis = node.attrs.get("axis",-1) % len(specs[node.inputs[0]].shape)
        return count*specs[node.inputs[0]].shape[axis]*itemsize,count
    return count*itemsize*len(node.inputs),count


def _generic_descriptor(node, specs, hw, optimize):
    out = specs[node.output]
    if node.op in {"softmax","layernorm"}:
        kind = "rows"
        iteration_size = out.numel // out.shape[-1]
    elif node.op == "sum":
        kind = "reduction"
        iteration_size = out.numel
    else:
        kind = "flat"
        iteration_size = out.numel
    target = 256 if optimize else 64
    step = min(iteration_size, target if kind != "rows" else (4 if optimize else 1))
    while step:
        input_bytes,count = _generic_sizes(node,specs,kind,0,step)
        if input_bytes <= hw.bank_bytes*3 and count*np.dtype(out.dtype).itemsize <= hw.bank_bytes:
            return {"kind":kind,"step":step,"iteration_size":iteration_size,"tile_elements":count}
        step //= 2
    raise ResourceError(f"simulator {node.op} requires a full reduction row that cannot fit SRAM; value={node.output}")


def _generic_flops(node,specs,count):
    if node.op == "sum":
        axis=node.attrs.get("axis",-1)%len(specs[node.inputs[0]].shape)
        return count*specs[node.inputs[0]].shape[axis]
    costs={"softmax":6,"layernorm":12,"gelu":20}
    if node.op == "fused_elementwise":
        return count*sum(costs.get(s["op"],1) for s in node.attrs["steps"])
    return count*costs.get(node.op,1)


def _validate_schedule(plan: SimulatorPlan):
    """Check dependencies, exclusive resources, SRAM contents and transfers."""
    hw=plan.hardware
    instructions=plan.instructions
    resource_intervals={}
    bank_intervals={}
    producers={n.output: [] for n in plan.graph.ordered_nodes()}
    specs=plan.graph.specs()
    coverage={name:[] for name in producers}
    for index,ins in enumerate(instructions):
        if ins.id != index or ins.op not in {"load","zero","mma","epilogue","store","load_group","vector","store_group"}:
            raise ResourceError("invalid simulator instruction id or opcode")
        if any(not isinstance(d,int) or d<0 or d>=index for d in ins.deps):
            raise ResourceError(f"instruction {index} has invalid dependency")
        if not isinstance(ins.start_cycle,int) or not isinstance(ins.end_cycle,int) or ins.start_cycle<0 or ins.end_cycle<=ins.start_cycle:
            raise ResourceError("instruction cycles must be nonnegative ordered integers")
        if any(instructions[d].end_cycle>ins.start_cycle for d in ins.deps):
            raise ResourceError(f"instruction {index} starts before its dependencies complete")
        transfer=ins.op in {"load","load_group","store","store_group"}
        prefix="dma" if transfer else "compute"
        limit=hw.transfer_units if transfer else hw.compute_units
        if ins.resource not in {f"{prefix}:{i}" for i in range(limit)}:
            raise ResourceError(f"instruction {index} uses nonexistent resource")
        work=ins.transfer_bytes if transfer else ins.flops
        if not isinstance(work,int) or work<0:
            raise ResourceError("instruction work must be a nonnegative integer")
        rate=hw.bytes_per_cycle_per_transfer if transfer else hw.flops_per_cycle_per_unit
        if ins.end_cycle-ins.start_cycle < hw.instruction_setup_cycles+math.ceil(work/rate):
            raise ResourceError("instruction duration is shorter than its modeled cost")
        resource_intervals.setdefault(ins.resource,[]).append(ins)
        if len(set(ins.banks)) != len(ins.banks) or any(not isinstance(b,int) or not 0<=b<hw.banks for b in ins.banks):
            raise ResourceError("instruction uses invalid SRAM bank")
        for b in ins.banks:
            bank_intervals.setdefault(b,[]).append(ins)
        for b,size in ins.attrs.get("size_by_bank",{}).items():
            if int(b) not in ins.banks or not isinstance(size,int) or not 0<=size<=hw.bank_bytes:
                raise ResourceError("instruction exceeds SRAM bank capacity")
        if ins.op in {"store","store_group"}:
            if ins.attrs.get("tensor") not in producers:
                raise ResourceError("store targets a nonexistent graph value")
            producers[ins.attrs["tensor"]].append(ins)
            name=ins.attrs["tensor"]
            shape=specs[name].shape
            if ins.op=="store":
                slices=ins.attrs.get("slices",())
                if len(shape)!=2 or len(slices)!=2 or any(len(s)!=2 or not all(isinstance(v,int) for v in s) or not 0<=s[0]<s[1]<=shape[axis] for axis,s in enumerate(slices)):
                    raise ResourceError("invalid matrix store region")
                (row0,row1),(col0,col1)=slices
                coverage[name].extend((row*shape[1]+col0,row*shape[1]+col1) for row in range(row0,row1))
            else:
                start,end=ins.attrs.get("start"),ins.attrs.get("end")
                factor=shape[-1] if ins.attrs.get("kind")=="rows" else 1
                if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=specs[name].numel//factor:
                    raise ResourceError("invalid vector store region")
                coverage[name].append((start*factor,end*factor))
    for name,regions in coverage.items():
        cursor=0
        for start,end in sorted(regions):
            if start!=cursor:
                raise ResourceError(f"stores overlap or leave an unwritten region of {name}")
            cursor=end
        if cursor!=specs[name].numel:
            raise ResourceError(f"stores do not produce all elements of {name}")
    for intervals in [*resource_intervals.values(),*bank_intervals.values()]:
        ordered=sorted(intervals,key=lambda i:(i.start_cycle,i.id))
        if any(a.end_cycle>b.start_cycle for a,b in zip(ordered,ordered[1:])):
            raise ResourceError("overlapping instructions contend for a resource or SRAM bank")
    tokens={}
    nodes=plan.scheduled_nodes
    if {n.output for n in nodes}!=set(producers) or len(nodes)!=len(producers):
        raise ResourceError("simulator node order does not match its graph")
    for ins in sorted(instructions,key=lambda i:(i.start_cycle,i.id)):
        for bank,token in ins.attrs.get("buffer_reads",{}).items():
            if int(bank) not in ins.banks or tokens.get(int(bank))!=token:
                raise ResourceError(f"instruction {ins.id} reads uninitialized or overwritten SRAM")
        for bank,token in ins.attrs.get("buffer_writes",{}).items():
            if int(bank) not in ins.banks:
                raise ResourceError("instruction writes an unreserved SRAM bank")
            tokens[int(bank)]=token
        reads=[]
        if ins.op=="load":
            reads=[ins.attrs["tensor"]]
        elif ins.op=="load_group":
            if not 0<=ins.attrs.get("node_index",-1)<len(nodes):
                raise ResourceError("vector instruction references invalid graph node")
            reads=list(nodes[ins.attrs["node_index"]].inputs)
        for value in reads:
            if value in producers and (not producers[value] or max(p.end_cycle for p in producers[value])>ins.start_cycle):
                raise ResourceError(f"instruction {ins.id} reads global tensor {value} before its producer completes")


def _gather_broadcast(array,shape,indices):
    coordinates=np.unravel_index(indices,shape)
    offset=len(shape)-array.ndim
    source=tuple(np.zeros_like(indices) if d==1 else coordinates[offset+i] for i,d in enumerate(array.shape))
    return np.ascontiguousarray(array[source])


def _load_group(node,values,specs,kind,start,end):
    output_shape=specs[node.output].shape
    if kind=="rows":
        x=values[node.inputs[0]].reshape(-1,output_shape[-1])[start:end].copy()
        return [x,*[values[v].copy() for v in node.inputs[1:]]]
    indices=np.arange(start,end)
    if kind=="reduction":
        x=values[node.inputs[0]]
        axis=node.attrs.get("axis",-1)%x.ndim
        coordinates=list(np.unravel_index(indices,output_shape))
        if node.attrs.get("keepdims",True):
            coordinates.pop(axis)
        coordinates.insert(axis,np.arange(x.shape[axis]))
        gather=tuple(c[None,:] if i==axis else c[:,None] for i,c in enumerate(coordinates))
        return [np.ascontiguousarray(x[gather])]
    if node.op=="transpose":
        axes=tuple(node.attrs.get("axes",reversed(range(len(output_shape)))))
        coords=np.unravel_index(indices,output_shape)
        source=[None]*len(axes)
        for out_axis,in_axis in enumerate(axes): source[in_axis]=coords[out_axis]
        return [np.ascontiguousarray(values[node.inputs[0]][tuple(source)])]
    if node.op in {"reshape","identity"}:
        return [values[node.inputs[0]].reshape(-1)[start:end].copy()]
    return [_gather_broadcast(values[v],output_shape,indices) for v in node.inputs]


def _activation(x,activation,dtype):
    x=x.astype(np.float32)
    if activation=="relu":
        x=np.maximum(x,0)
    elif activation=="gelu":
        x=np.float32(.5)*x*(np.float32(1)+np.tanh(np.float32(math.sqrt(2/math.pi))*(x+np.float32(.044715)*x*x*x)))
    elif activation!="none":
        raise ExecutionError(f"unsupported simulated activation {activation}")
    return x.astype(dtype)


def _vector(node,payload,dtype):
    def element(op,inputs):
        values=[x.astype(np.float32) for x in inputs]
        if op=="add": return (values[0]+values[1]).astype(dtype)
        if op=="mul": return (values[0]*values[1]).astype(dtype)
        if op in {"relu","gelu"}: return _activation(values[0],op,dtype)
        if op=="identity": return inputs[0].copy()
        raise ExecutionError(f"unsupported simulated elementwise instruction {op}")
    if node.op in {"add","mul","relu","gelu"}:
        return element(node.op,payload)
    if node.op in {"reshape","transpose","identity"}:
        return payload[0].copy()
    if node.op=="fused_elementwise":
        local=dict(zip(node.inputs,payload))
        for step in node.attrs["steps"]:
            local[step["output"]]=element(step["op"],[local[v] for v in step["inputs"]])
        return local[node.output]
    x=payload[0].astype(np.float32)
    if node.op=="sum":
        return np.sum(x,axis=-1,dtype=np.float32).astype(dtype)
    if node.op=="softmax":
        exp=np.exp(x-np.max(x,axis=-1,keepdims=True))
        return (exp/np.sum(exp,axis=-1,keepdims=True,dtype=np.float32)).astype(dtype)
    if node.op=="layernorm":
        mean=np.mean(x,axis=-1,keepdims=True,dtype=np.float32)
        centered=x-mean
        variance=np.mean(centered*centered,axis=-1,keepdims=True,dtype=np.float32)
        normalized=centered/np.sqrt(variance+np.float32(node.attrs.get("eps",1e-5)))
        return (normalized*payload[1].astype(np.float32)+payload[2].astype(np.float32)).astype(dtype)
    raise ExecutionError(f"unsupported simulated vector instruction {node.op}")


def simulate(plan: SimulatorPlan, inputs: Mapping[str,np.ndarray]) -> dict[str,np.ndarray]:
    """Execute the emitted instruction stream; no whole-graph reference fallback."""
    _validate_schedule(plan)
    graph=plan.graph
    graph.validate()
    specs=graph.specs()
    if set(inputs)!=set(graph.inputs):
        raise ExecutionError(f"simulator input names must be {sorted(graph.inputs)}")
    values={}
    for name,spec in graph.inputs.items():
        array=inputs[name]
        if not isinstance(array,np.ndarray) or array.shape!=spec.shape or str(array.dtype)!=spec.dtype or not array.flags.c_contiguous:
            raise ExecutionError(f"simulator input {name} requires contiguous {spec.shape} {spec.dtype}")
        values[name]=array.copy()
    values.update({name:value.copy() for name,value in graph.constants.items()})
    nodes=plan.scheduled_nodes
    for node in nodes:
        spec=specs[node.output]
        values[node.output]=np.empty(spec.shape,dtype=spec.dtype)
    scratch={}
    for ins in sorted(plan.instructions,key=lambda i:(i.start_cycle,i.id)):
        attrs=ins.attrs
        if ins.op=="zero":
            scratch[attrs["bank"]]=np.zeros(attrs["shape"],dtype=np.float32)
        elif ins.op=="load":
            source=values[attrs["tensor"]]
            if "broadcast_shape" in attrs:
                source=np.broadcast_to(source,attrs["broadcast_shape"])
            scratch[attrs["bank"]]=source[tuple(slice(*s) for s in attrs["slices"])].copy()
        elif ins.op=="mma":
            a,b=scratch[attrs["a_bank"]],scratch[attrs["b_bank"]]
            scratch[attrs["c_bank"]]+=a.astype(np.float32)@b.astype(np.float32)
        elif ins.op=="epilogue":
            x=scratch[attrs["c_bank"]].astype(attrs["dtype"])
            if attrs["bias_bank"] is not None:
                x=(x.astype(np.float32)+scratch[attrs["bias_bank"]].astype(np.float32)).astype(attrs["dtype"])
            scratch[attrs["c_bank"]]=_activation(x,attrs["activation"],attrs["dtype"])
        elif ins.op=="store":
            values[attrs["tensor"]][tuple(slice(*s) for s in attrs["slices"])]=scratch[attrs["bank"]]
        elif ins.op=="load_group":
            node=nodes[attrs["node_index"]]
            scratch[attrs["payload_bank"]]=_load_group(node,values,specs,attrs["kind"],attrs["start"],attrs["end"])
        elif ins.op=="vector":
            node=nodes[attrs["node_index"]]
            scratch[attrs["output_bank"]]=_vector(node,scratch[attrs["payload_bank"]],specs[node.output].dtype)
        elif ins.op=="store_group":
            target=values[attrs["tensor"]]
            target=target.reshape(-1,target.shape[-1]) if attrs["kind"]=="rows" else target.reshape(-1)
            target[attrs["start"]:attrs["end"]]=scratch[attrs["bank"]]
    return {name:values[name].copy() for name in graph.outputs}


def estimate(plan: SimulatorPlan) -> dict[str,Any]:
    """Synthetic cycle/resource estimates. These are never GPU measurements."""
    _validate_schedule(plan)
    cycles=max((i.end_cycle for i in plan.instructions),default=0)
    bank_sizes={}
    peak=0
    busy={}
    for ins in sorted(plan.instructions,key=lambda i:(i.start_cycle,i.id)):
        bank_sizes.update({int(k):v for k,v in ins.attrs.get("size_by_bank",{}).items()})
        peak=max(peak,sum(bank_sizes.values()))
        busy[ins.resource]=busy.get(ins.resource,0)+ins.end_cycle-ins.start_cycle
    specs=plan.graph.specs()
    return {"kind":"simulated_estimate_not_hardware_measurement",
            "cycles":cycles,"estimated_microseconds":cycles/(plan.hardware.core_clock_ghz*1000),
            "instruction_count":len(plan.instructions),
            "transfer_bytes":sum(i.transfer_bytes for i in plan.instructions),
            "modeled_flops":sum(i.flops for i in plan.instructions),
            "peak_local_bytes":peak,
            "global_storage_bytes":sum(s.nbytes for s in specs.values()),
            "resource_busy_cycles":busy,
            "resource_utilization":{r:(v/cycles if cycles else 0) for r,v in busy.items()}}
