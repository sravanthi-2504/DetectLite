import torch
from models.detector import MobileViTDetector

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = MobileViTDetector(num_classes=20).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    print("Device:", device)
    print(f"Parameters: {params:,} ({params/1e6:.3f} M)")
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, 3, 224, 224, device=device)
        flops = FlopCountAnalysis(model, x).total()
        print(f"FLOPs: {flops:,} ({flops/1e9:.3f} GFLOPs)")
    except Exception as e:
        print("FLOP profiling unavailable:", type(e).__name__, e)

if __name__ == "__main__":
    main()
