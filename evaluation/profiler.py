import time
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count


def profile_model(model, device="cpu", image_size=224, warmup=5, iterations=20):
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)

    if device == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    latency_ms = elapsed / iterations * 1000.0
    fps = 1000.0 / latency_ms

    params = sum(p.numel() for p in model.parameters())

    # FLOP counting can fail for unsupported custom operators; keep it optional.
    try:
        flops = FlopCountAnalysis(model, x).total()
    except Exception:
        flops = None

    return {
        "parameters": params,
        "flops": flops,
        "latency_ms": latency_ms,
        "fps": fps,
    }
