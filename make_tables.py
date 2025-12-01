import json
import argparse


def to_markdown(results: dict) -> str:
    lines = [
        "| Model | PLCC (KonIQ) | SRCC (KonIQ) |",
        "|---|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m.get('plcc', 'NA'):.3f} | {m.get('srcc', 'NA'):.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results/eval_results.json")
    args = ap.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(to_markdown(results))


if __name__ == "__main__":
    main()
