"""Report actual GPU backend readiness; never substitute simulated results."""

import json
import platform
import subprocess
import sys


def main():
    info = {"python": sys.version, "platform": platform.platform(), "available": False}
    try:
        from mlgpu.gpu import ensure_available
        ensure_available()
        import torch
        import triton
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update({"available": True, "torch": torch.__version__, "triton": triton.__version__,
                     "cuda_runtime": torch.version.cuda, "gpu": props.name,
                     "compute_capability": list(torch.cuda.get_device_capability(device)),
                     "total_memory_bytes": props.total_memory,
                     "multiprocessor_count": props.multi_processor_count,
                     "shared_memory_per_block": props.shared_memory_per_block})
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,power.limit,temperature.gpu",
                                     "--format=csv,noheader"], capture_output=True, text=True, check=True)
            info["nvidia_smi_driver_power_temperature"] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            info["nvidia_smi_error"] = str(exc)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(info, indent=2))
    return 0 if info["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
