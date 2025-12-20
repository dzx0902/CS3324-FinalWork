import json
from src.models.iqa_heads.resnet_baseline import ResNetBaselineIQA
from src.models.iqa_heads.tiny_iqa import TinyIQA_R18, TinyIQA_R34, TinyIQA_R18_MS
from src.utils.model_stats import count_trainable_params, benchmark_inference


def run():
    models = {
        "resnet_baseline": ResNetBaselineIQA(),
        "tiny_r18": TinyIQA_R18(),
        "tiny_r34": TinyIQA_R34(),
        "tiny_r18_ms": TinyIQA_R18_MS(),
    }
    stats = {}
    for name, m in models.items():
        params = count_trainable_params(m)
        ms = benchmark_inference(m, input_size=(1, 3, 224, 224), device="cuda", n_warmup=5, n_iters=20)
        stats[name] = {"trainable_params": int(params), "avg_ms_per_image": float(ms)}
    with open("results/model_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    run()
