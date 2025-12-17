import argparse
import json
import os


def to_markdown(results, out_path):
    lines = []
    models = list(results.keys())
    datasets = list(next(iter(results.values())).keys())
    header = "| Model | " + " | ".join(datasets) + " |"
    sep = "|" + "---|" * (len(datasets) + 1)
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells = [m]
        for d in datasets:
            r = results[m][d]
            cells.append(f"PLCC {r['plcc']:.3f}, SRCC {r['srcc']:.3f}, RMSE {r['rmse']:.3f}, MAE {r['mae']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def to_latex(results, out_path):
    models = list(results.keys())
    datasets = list(next(iter(results.values())).keys())
    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append("\\hline")
    header = "Model & " + " & ".join(datasets) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for m in models:
        cells = [m]
        for d in datasets:
            r = results[m][d]
            cells.append(f"{r['plcc']:.3f}/{r['srcc']:.3f}/{r['rmse']:.3f}/{r['mae']:.3f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/eval_results.json")
    parser.add_argument("--out_md", type=str, default="results/table.md")
    parser.add_argument("--out_tex", type=str, default="results/table.tex")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    to_markdown(results, args.out_md)
    to_latex(results, args.out_tex)


if __name__ == "__main__":
    main()

