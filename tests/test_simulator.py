from dataclasses import replace
import json

import numpy as np
import pytest

from mlgpu.errors import ExecutionError, ResourceError
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute
from mlgpu.simulator import HardwareConfig, compile_simulator, estimate, simulate


def _inputs(graph,seed=19):
    rng=np.random.default_rng(seed)
    return {name:np.ascontiguousarray((rng.standard_normal(spec.shape)*.3).astype(spec.dtype))
            for name,spec in graph.inputs.items()}


def _bias_graph(dtype="float32",m=13,k=17,n=11):
    return Graph({"x":TensorSpec((m,k),dtype),"w":TensorSpec((k,n),dtype),
                  "b":TensorSpec((n,),dtype)},
                 [Node("matmul",("x","w"),"mm"),Node("add",("mm","b"),"biased"),
                  Node("gelu",("biased",),"y")],("y",),name="matrix_bias_gelu")


def _assert_reference(graph,optimize=True,hardware=None):
    inputs=_inputs(graph)
    plan=compile_simulator(graph,hardware=hardware,optimize=optimize)
    actual=simulate(plan,inputs)
    expected=execute(graph,inputs)
    for name in graph.outputs:
        assert actual[name].dtype==expected[name].dtype
        np.testing.assert_allclose(actual[name],expected[name],rtol=4e-3 if actual[name].dtype==np.float16 else 2e-4,
                                   atol=3e-3 if actual[name].dtype==np.float16 else 3e-5)
    return plan


@pytest.mark.parametrize("dtype",["float16","float32"])
@pytest.mark.parametrize("optimize",[False,True])
def test_tiled_matmul_bias_gelu_irregular_shapes(dtype,optimize):
    plan=_assert_reference(_bias_graph(dtype),optimize)
    assert {"load","mma","store","zero","epilogue"}<={i.op for i in plan.instructions}


@pytest.mark.parametrize("dtype",["float16","float32"])
def test_multilayer_graph(dtype):
    graph=Graph({"x":TensorSpec((7,13),dtype),"w1":TensorSpec((13,19),dtype),
                 "b1":TensorSpec((19,),dtype),"w2":TensorSpec((19,9),dtype),"b2":TensorSpec((9,),dtype)},
                [Node("matmul",("x","w1"),"a"),Node("add",("a","b1"),"ab"),
                 Node("gelu",("ab",),"h"),Node("matmul",("h","w2"),"c"),
                 Node("add",("c","b2"),"y")],("y",),name="mlp")
    _assert_reference(graph)


@pytest.mark.parametrize("dtype",["float16","float32"])
def test_layernorm_residual(dtype):
    graph=Graph({"x":TensorSpec((2,3,17),dtype),"scale":TensorSpec((17,),dtype),
                 "bias":TensorSpec((17,),dtype),"residual":TensorSpec((2,3,17),dtype)},
                [Node("layernorm",("x","scale","bias"),"norm",{"eps":1e-5}),
                 Node("add",("norm","residual"),"y")],("y",))
    _assert_reference(graph)


@pytest.mark.parametrize("optimize",[False,True])
def test_broadcast_elementwise_branch_and_multiple_outputs(optimize):
    graph=Graph({"x":TensorSpec((3,1,7)),"b":TensorSpec((1,5,1)),"scale":TensorSpec((7,))},
                [Node("add",("x","b"),"a"),Node("relu",("a",),"r"),
                 Node("mul",("r","scale"),"m"),Node("gelu",("m",),"y")],("r","y"))
    _assert_reference(graph,optimize)


def test_transpose_reshape_softmax():
    graph=Graph({"x":TensorSpec((2,3,7))},
                [Node("transpose",("x",),"t",{"axes":(1,0,2)}),
                 Node("reshape",("t",),"r",{"shape":(6,7)}),
                 Node("softmax",("r",),"y")],("t","y"))
    _assert_reference(graph)


@pytest.mark.parametrize("axis",[0,1,2,-1])
@pytest.mark.parametrize("keepdims",[False,True])
def test_sum_axis_tiles(axis,keepdims):
    graph=Graph({"x":TensorSpec((3,5,7))},
                [Node("sum",("x",),"y",{"axis":axis,"keepdims":keepdims})],("y",))
    _assert_reference(graph)


def test_stable_reductions_large_values_and_constant_rows():
    graph=Graph({"x":TensorSpec((3,17)),"w":TensorSpec((17,)),"b":TensorSpec((17,))},
                [Node("softmax",("x",),"s"),Node("layernorm",("x","w","b"),"l")],("s","l"))
    values={"x":np.stack([np.full(17,1000,dtype=np.float32),np.linspace(-1000,1000,17,dtype=np.float32),
                          np.linspace(-1,1,17,dtype=np.float32)]),
            "w":np.ones(17,dtype=np.float32),"b":np.zeros(17,dtype=np.float32)}
    actual=simulate(compile_simulator(graph),values)
    expected=execute(graph,values)
    for name in graph.outputs:
        assert np.all(np.isfinite(actual[name]))
        np.testing.assert_allclose(actual[name],expected[name],rtol=1e-5,atol=2e-6)


def test_optimized_schedule_has_measured_model_effect():
    graph=_bias_graph(m=37,k=41,n=29)
    baseline=estimate(_assert_reference(graph,optimize=False))
    optimized=estimate(_assert_reference(graph,optimize=True))
    assert optimized["instruction_count"]<baseline["instruction_count"]
    assert optimized["transfer_bytes"]<baseline["transfer_bytes"]
    assert optimized["cycles"]<baseline["cycles"]
    assert optimized["kind"]=="simulated_estimate_not_hardware_measurement"


def test_scheduler_overlaps_independent_dma_and_compute():
    graph=Graph({"x":TensorSpec((65,73)),"w":TensorSpec((73,49))},
                [Node("matmul",("x","w"),"y")],("y",))
    plan=_assert_reference(graph)
    transfers=[i for i in plan.instructions if i.resource.startswith("dma")]
    compute=[i for i in plan.instructions if i.op=="mma"]
    assert any(max(a.start_cycle,b.start_cycle)<min(a.end_cycle,b.end_cycle)
               for a in transfers for b in compute)


def test_small_local_memory_retiles_instead_of_overcommitting():
    hw=HardwareConfig(local_memory_bytes=128,banks=4)
    plan=_assert_reference(_bias_graph(m=3,k=5,n=7),hardware=hw)
    assert plan.tiles[next(n.output for n in plan.scheduled_nodes if n.op in {"matmul","fused_matmul"})]["block_m"]<=2
    assert estimate(plan)["peak_local_bytes"]<=hw.local_memory_bytes


def test_full_row_requirement_fails_explicitly():
    graph=Graph({"x":TensorSpec((2,4097))},[Node("softmax",("x",),"y")],("y",))
    with pytest.raises(ResourceError,match="cannot fit SRAM"):
        compile_simulator(graph)


@pytest.mark.parametrize("kwargs",[{"banks":3},{"local_memory_bytes":0},{"compute_units":True},
                                   {"transfer_units":-1},{"core_clock_ghz":float("nan")},
                                   {"instruction_setup_cycles":0},{"max_tile":1.5}])
def test_invalid_hardware_rejected(kwargs):
    with pytest.raises(ResourceError): HardwareConfig(**kwargs)


def test_estimate_and_schedule_are_inspectable_json():
    plan=compile_simulator(_bias_graph())
    report=estimate(plan)
    assert 0<report["peak_local_bytes"]<=plan.hardware.local_memory_bytes
    assert all(0<=u<=1 for u in report["resource_utilization"].values())
    assert report["estimated_microseconds"]==pytest.approx(report["cycles"]/1000)
    document=json.loads(plan.format())
    assert document["backend"]=="synthetic_tensor_accelerator"
    assert document["instructions"]


def _tamper(plan,index,**changes):
    instructions=list(plan.instructions)
    instructions[index]=replace(instructions[index],**changes)
    return replace(plan,instructions=tuple(instructions))


def test_schedule_dependency_violation_rejected():
    plan=compile_simulator(_bias_graph())
    index=next(i.id for i in plan.instructions if i.op=="mma")
    invalid=_tamper(plan,index,start_cycle=0,end_cycle=100)
    with pytest.raises(ResourceError): simulate(invalid,_inputs(plan.graph))


def test_schedule_invalid_bank_rejected():
    plan=compile_simulator(_bias_graph())
    with pytest.raises(ResourceError,match="bank"):
        estimate(_tamper(plan,0,banks=(plan.hardware.banks,)))


def test_schedule_sram_overflow_rejected():
    plan=compile_simulator(_bias_graph())
    attrs={**plan.instructions[0].attrs,"size_by_bank":{"2":plan.hardware.bank_bytes+1}}
    with pytest.raises(ResourceError,match="capacity"):
        estimate(_tamper(plan,0,attrs=attrs))


def test_schedule_reads_overwritten_sram_rejected():
    plan=compile_simulator(_bias_graph())
    index=next(i.id for i in plan.instructions if i.op=="mma")
    attrs=dict(plan.instructions[index].attrs)
    attrs["buffer_reads"]={**attrs["buffer_reads"],str(attrs["a_bank"]):"wrong-tile"}
    with pytest.raises(ResourceError,match="overwritten SRAM"):
        estimate(_tamper(plan,index,attrs=attrs))


def test_schedule_missing_store_rejected():
    plan=compile_simulator(_bias_graph())
    invalid=replace(plan,instructions=plan.instructions[:-1])
    with pytest.raises(ResourceError,match="all elements"):
        estimate(invalid)


def test_bad_inputs_fail_before_execution():
    graph=_bias_graph()
    plan=compile_simulator(graph)
    values=_inputs(graph)
    with pytest.raises(ExecutionError,match="names"): simulate(plan,{})
    values["x"]=values["x"].astype(np.float16)
    with pytest.raises(ExecutionError,match="input x"): simulate(plan,values)


def test_input_and_constant_graph_outputs():
    graph=Graph({"x":TensorSpec((2,3))},[],("x","c"),{"c":np.ones((3,),dtype=np.float32)})
    actual=simulate(compile_simulator(graph),{"x":np.zeros((2,3),dtype=np.float32)})
    np.testing.assert_array_equal(actual["x"],0)
    np.testing.assert_array_equal(actual["c"],1)
