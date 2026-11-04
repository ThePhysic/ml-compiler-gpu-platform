"""Independent framework baseline, never used by the custom GPU runtime."""


def make_callable(graph, inputs):
    import torch
    import torch.nn.functional as F
    graph.validate()
    nodes = graph.ordered_nodes()
    names = tuple(graph.inputs)
    device = next(iter(inputs.values())).device if inputs else torch.device("cuda:0")
    constants = {name: torch.from_numpy(a.copy()).to(device) for name, a in graph.constants.items()}

    def run(*args):
        values = dict(constants)
        values.update(zip(names, args))
        for node in nodes:
            a = [values[x] for x in node.inputs]
            op = node.op
            attrs = node.attrs
            if op == "matmul":
                out = a[0] @ a[1]
            elif op == "add":
                out = a[0] + a[1]
            elif op == "mul":
                out = a[0] * a[1]
            elif op == "relu":
                out = torch.relu(a[0])
            elif op == "gelu":
                out = F.gelu(a[0], approximate="tanh")
            elif op == "layernorm":
                out = F.layer_norm(a[0], (a[0].shape[-1],), a[1], a[2], attrs.get("eps", 1e-5))
            elif op == "softmax":
                out = torch.softmax(a[0], dim=-1)
            elif op == "sum":
                out = a[0].sum(dim=attrs.get("axis", -1), keepdim=attrs.get("keepdims", True), dtype=torch.float32).to(a[0].dtype)
            elif op == "reshape":
                out = a[0].reshape(attrs["shape"]).clone()
            elif op == "transpose":
                axes = attrs.get("axes", tuple(reversed(range(a[0].ndim))))
                out = a[0].permute(axes).contiguous()
            elif op == "identity":
                out = a[0].clone()
            else:
                raise ValueError(f"framework baseline requires unfused supported graph: {op}")
            values[node.output] = out
        return tuple(values[name] for name in graph.outputs)
    return run, tuple(inputs[name] for name in names)
