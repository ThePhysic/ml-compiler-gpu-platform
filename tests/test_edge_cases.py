"""Independent boundary and randomized DAG regression tests."""
import copy

import numpy as np
import pytest

from mlgpu.errors import GraphError
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute
from mlgpu.runtime import Program
from mlgpu.simulator import compile_simulator, simulate


@pytest.mark.parametrize("steps",[
    [{"op":"relu","inputs":None,"output":"y"}],
    [{"op":[],"inputs":["x"],"output":"y"}],
    [{"op":"relu","inputs":["x"],"output":[]}],
    [{"op":"relu","inputs":["x"],"output":1},
     {"op":"relu","inputs":[1],"output":"y"}],
    [{"op":"relu","inputs":["x"],"output":""},
     {"op":"relu","inputs":[""],"output":"y"}],
    [{"op":"relu","inputs":[[]],"output":"y"}],
    [{"op":"relu","inputs":["x"],"output":None},
     {"op":"relu","inputs":[None],"output":"y"}],
])
def test_malformed_fused_region_has_graph_error(steps):
    graph=Graph({"x":TensorSpec((2,3))},
                [Node("fused_elementwise",("x",),"y",{"steps":steps})],("y",))
    with pytest.raises(GraphError): graph.validate()


def test_malformed_fused_activation_has_graph_error():
    graph=Graph({"a":TensorSpec((2,3)),"b":TensorSpec((3,4))},
                [Node("fused_matmul",("a","b"),"y",{"activation":[]})],("y",))
    with pytest.raises(GraphError): graph.validate()


def _random_graph(seed,dtype):
    rng=np.random.default_rng(seed)
    graph=Graph({"x":TensorSpec((3,7),dtype),"z":TensorSpec((3,7),dtype),
                 "b":TensorSpec((7,),dtype)},[],(),
                {"c":np.full((1,7),.125,dtype=dtype)},name=f"random_dag_{seed}")
    available=["x","z"]
    for index in range(24):
        op=rng.choice(["add","mul","relu","gelu"]).item()
        first=rng.choice(available).item()
        if op in {"add","mul"}:
            # Repeated operands deliberately exercise use counts and aliasing.
            second=first if index%4==0 else rng.choice([*available,"b","c"]).item()
            inputs=(first,second)
        else:
            inputs=(first,)
        output=f"v{index}"
        graph.nodes.append(Node(op,inputs,output))
        available.append(output)
    graph.outputs=("v23","v8","x","c")
    rng.shuffle(graph.nodes)
    inputs={name:np.ascontiguousarray((rng.normal(size=spec.shape)*.15).astype(dtype))
            for name,spec in graph.inputs.items()}
    return graph,inputs


@pytest.mark.parametrize("dtype",["float16","float32"])
@pytest.mark.parametrize("seed",range(12))
def test_random_dag_reference_runtime_and_simulator(dtype,seed):
    graph,inputs=_random_graph(seed,dtype)
    before=copy.deepcopy(graph.to_dict())
    expected=execute(graph,inputs)
    for optimize in (False,True):
        program=Program(graph,optimize=optimize)
        actual=program.run(inputs)
        simulated=simulate(compile_simulator(graph,optimize=optimize),inputs)
        for name in graph.outputs:
            np.testing.assert_array_equal(actual[name],expected[name])
            np.testing.assert_allclose(simulated[name],expected[name],rtol=1e-5,atol=1e-6)
        allocations=list(program.plan.memory.allocations.values())
        for index,a in enumerate(allocations):
            for b in allocations[index+1:]:
                storage_overlap=max(a.offset,b.offset)<min(a.offset+a.reserved_size,b.offset+b.reserved_size)
                if storage_overlap:
                    assert a.end<b.start or b.end<a.start
    assert graph.to_dict()==before


def test_fused_elementwise_preserves_fp16_materialization_rounding():
    graph=Graph({"x":TensorSpec((1,),"float16"),"b":TensorSpec((1,),"float16"),
                 "c":TensorSpec((1,),"float16")},
                [Node("add",("x","b"),"a"),Node("add",("a","c"),"y")],("y",))
    inputs={"x":np.array([2048],np.float16),"b":np.array([1],np.float16),"c":np.array([-2048],np.float16)}
    program=Program(graph)
    assert program.plan.nodes[0].op=="fused_elementwise"
    # Retaining the intermediate in FP32 would incorrectly produce 1.
    np.testing.assert_array_equal(program.run(inputs)["y"],[0])
    np.testing.assert_array_equal(simulate(compile_simulator(graph),inputs)["y"],[0])


def test_fused_epilogue_preserves_fp16_matmul_boundary_rounding():
    graph=Graph({"a":TensorSpec((1,2),"float16"),"w":TensorSpec((2,1),"float16"),
                 "b":TensorSpec((1,),"float16")},
                [Node("matmul",("a","w"),"m"),Node("add",("m","b"),"z"),
                 Node("relu",("z",),"y")],("y",))
    inputs={"a":np.array([[2048,1]],np.float16),"w":np.ones((2,1),np.float16),"b":np.array([-2048],np.float16)}
    program=Program(graph)
    assert program.plan.nodes[0].op=="fused_matmul"
    np.testing.assert_array_equal(program.run(inputs)["y"],[[0]])
    np.testing.assert_array_equal(simulate(compile_simulator(graph),inputs)["y"],[[0]])


def test_compilation_snapshots_constants_and_bindings_preserve_external_outputs():
    graph=Graph({"x":TensorSpec((2,3))},[Node("add",("x","c"),"a"),
                Node("add",("c","c"),"folded"),Node("add",("a","folded"),"y")],
                ("x","c","folded","y"),{"c":np.full((3,),2,dtype=np.float32)})
    values={"x":np.ones((2,3),dtype=np.float32)}
    program=Program(graph)
    graph.constants["c"].fill(-99)
    bound=program.bind(values)
    first=bound.run()
    np.testing.assert_array_equal(first["c"],2)
    np.testing.assert_array_equal(first["folded"],4)
    np.testing.assert_array_equal(first["y"],7)
    first["c"].fill(99)
    first["x"].fill(99)
    second=bound.run()
    np.testing.assert_array_equal(second["c"],2)
    np.testing.assert_array_equal(second["x"],1)


def test_fusion_respects_gpu_external_operand_limit():
    from mlgpu.gpu import MAX_REGION_INPUTS

    # Sixteen binary instructions need seventeen distinct external values.
    graph=Graph({f"x{i}":TensorSpec((2,3)) for i in range(17)},[],("v15",))
    previous="x0"
    for index in range(16):
        output=f"v{index}"
        graph.nodes.append(Node("add",(previous,f"x{index+1}"),output))
        previous=output
    program=Program(graph)
    regions=[node for node in program.plan.nodes if node.op=="fused_elementwise"]
    assert regions
    assert all(len(node.inputs)<=MAX_REGION_INPUTS for node in regions)
    inputs={name:np.full((2,3),index,dtype=np.float32) for index,name in enumerate(graph.inputs)}
    np.testing.assert_array_equal(program.run(inputs)["v15"],136)


