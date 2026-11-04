"""Independent simulator audit: instruction metadata must describe actual work."""
from dataclasses import replace

import numpy as np
import pytest

from mlgpu.errors import ExecutionError, ResourceError
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.simulator import HardwareConfig, compile_simulator, estimate, simulate


def _matrix_plan():
    graph = Graph({"a": TensorSpec((2, 2)), "b": TensorSpec((2, 2))},
                  [Node("matmul", ("a", "b"), "y")], ("y",))
    inputs = {"a": np.array([[1, 2], [3, 4]], np.float32),
              "b": np.array([[5, 6], [7, 8]], np.float32)}
    return compile_simulator(graph), inputs


def _change(plan, index, attrs):
    instructions = list(plan.instructions)
    instructions[index] = replace(instructions[index], attrs=attrs)
    return replace(plan, instructions=tuple(instructions))


def test_matrix_operand_banks_must_match_declared_sram_reads():
    plan, inputs = _matrix_plan()
    ins = next(i for i in plan.instructions if i.op == "mma")
    # This changes A@B into B@B while all token and reservation declarations
    # still claim the old instruction. Validation must reject that mismatch.
    invalid = _change(plan, ins.id, {**ins.attrs, "a_bank": ins.attrs["b_bank"]})
    with pytest.raises(ResourceError):
        simulate(invalid, inputs)


def test_load_sram_size_cannot_understate_actual_tensor_tile():
    plan, _ = _matrix_plan()
    ins = next(i for i in plan.instructions if i.op == "load")
    invalid = _change(plan, ins.id, {**ins.attrs, "size_by_bank": {str(ins.attrs["bank"]): 0}})
    with pytest.raises(ResourceError):
        estimate(invalid)


def test_instruction_semantics_must_match_plan_graph():
    graph = Graph({"x": TensorSpec((3,))}, [Node("relu", ("x",), "y")], ("y",))
    plan = compile_simulator(graph)
    invalid = replace(plan, scheduled_nodes=(Node("gelu", ("x",), "y"),))
    with pytest.raises(ResourceError):
        simulate(invalid, {"x": np.array([-1, 1, 2], np.float32)})


@pytest.mark.parametrize("field,value", [("tensor","a"), ("slices",[[0,2],[0,1]]),
                                         ("depth",[1,1]), ("role","bias")])
def test_matrix_load_attributes_must_match_actual_graph_operand(field,value):
    plan, _ = _matrix_plan()
    ins = next(i for i in plan.instructions if i.op == "load" and i.attrs["role"] == "b")
    invalid = _change(plan,ins.id,{**ins.attrs,field:value})
    with pytest.raises(ResourceError):
        estimate(invalid)


def test_epilogue_cannot_skip_reduction_instructions():
    plan, _ = _matrix_plan()
    ins = next(i for i in plan.instructions if i.op == "mma")
    invalid = _change(plan,ins.id,{**ins.attrs,"depth":[1,1]})
    with pytest.raises(ResourceError):
        estimate(invalid)


def test_dma_byte_count_cannot_understate_tile_transfer():
    plan, _ = _matrix_plan()
    instructions = list(plan.instructions)
    ins = next(i for i in instructions if i.op == "load")
    instructions[ins.id] = replace(ins,transfer_bytes=0)
    with pytest.raises(ResourceError):
        estimate(replace(plan,instructions=tuple(instructions)))


def test_missing_opcode_fields_report_resource_error():
    plan, _ = _matrix_plan()
    ins = next(i for i in plan.instructions if i.op == "mma")
    attrs = {key:value for key,value in ins.attrs.items() if key != "c_bank"}
    with pytest.raises(ResourceError):
        estimate(_change(plan,ins.id,attrs))


def test_sram_reuse_dependencies_must_be_explicit():
    plan, _ = _matrix_plan()
    instructions = list(plan.instructions)
    ins = next(i for i in instructions if i.op == "mma")
    # Timestamps remain valid, but removing dependency edges would make a future
    # reschedule unsafe. The declared instruction DAG must express the hazards.
    instructions[ins.id] = replace(ins,deps=())
    with pytest.raises(ResourceError):
        estimate(replace(plan,instructions=tuple(instructions)))


@pytest.mark.parametrize("width,local_bytes", [(513,65536),(2049,131072)])
def test_layernorm_large_constant_rows_are_exactly_zero(width,local_bytes):
    graph = Graph({"x":TensorSpec((2,width)),"w":TensorSpec((width,)),"b":TensorSpec((width,))},
                  [Node("layernorm",("x","w","b"),"y")],("y",))
    values = {"x":np.full((2,width),16777215,np.float32),"w":np.ones(width,np.float32),
              "b":np.zeros(width,np.float32)}
    plan = compile_simulator(graph,HardwareConfig(local_memory_bytes=local_bytes))
    np.testing.assert_array_equal(simulate(plan,values)["y"],0)


@pytest.mark.parametrize("inputs", [None,[],{"a":np.zeros((2,2),np.float32),1:np.zeros((2,2),np.float32)}])
def test_malformed_inputs_use_subsystem_error(inputs):
    plan, _ = _matrix_plan()
    with pytest.raises(ExecutionError):
        simulate(plan,inputs)
def test_inspected_simulator_tiles_are_detached_from_live_plan():
    graph = Graph({"a": TensorSpec((2, 3)), "b": TensorSpec((3, 4))}, [
        Node("matmul", ("a", "b"), "y")], ("y",))
    plan = compile_simulator(graph)
    inspected = plan.to_dict()
    original = plan.tiles["y"]["block_m"]
    inspected["tiles"]["y"]["block_m"] = 0
    assert plan.tiles["y"]["block_m"] == original
    assert estimate(plan)["instruction_count"] > 0
