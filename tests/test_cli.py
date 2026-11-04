import json

from mlgpu.cli import main


def test_inspect_command(capsys):
    assert main(["inspect", "--workload", "epilogue"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schedule"][0]["op"] == "fused_matmul"


def test_reference_demo_command(capsys):
    assert main(["demo", "--workload", "mlp", "--dtype", "float16"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["backends"]["reference"]["correct"]


def test_cli_graph_errors_nonzero(capsys):
    assert main(["demo", "--m", "0"]) == 1
    assert "GraphError" in capsys.readouterr().err
