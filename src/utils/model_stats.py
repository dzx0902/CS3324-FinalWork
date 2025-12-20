import time
import torch


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(model, input_size=(1, 3, 224, 224), device="cuda", n_warmup=5, n_iters=20):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()
    x = torch.randn(*input_size, device=dev)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.time()
        for _ in range(n_iters):
            _ = model(x)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t1 = time.time()
    avg = (t1 - t0) * 1000.0 / n_iters
    return avg
