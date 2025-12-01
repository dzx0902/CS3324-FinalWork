import csv
import sys
import os
from statistics import mean


def read_csv(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'Data/metas/koniq10k_scores_and_distributions.csv'
    if not os.path.isfile(path):
        print('not_found', path)
        return
    rows = read_csv(path)
    mos = [to_float(r.get('MOS')) for r in rows]
    z = [to_float(r.get('MOS_zscore')) for r in rows]
    mos = [v for v in mos if v is not None]
    z = [v for v in z if v is not None]
    def stats(arr):
        return {'min': min(arr), 'max': max(arr), 'mean': mean(arr)}
    s_mos = stats(mos)
    s_z = stats(z)
    def corr(a, b):
        import math
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        ma = sum(a) / n
        mb = sum(b) / n
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
        sa = math.sqrt(sum((x - ma) ** 2 for x in a) / n) + 1e-8
        sb = math.sqrt(sum((x - mb) ** 2 for x in b) / n) + 1e-8
        return cov / (sa * sb)
    print('rows', len(rows))
    print('MOS', s_mos)
    print('MOS_zscore', s_z)
    print('corr(MOS, MOS_zscore)', corr(mos, z))


if __name__ == '__main__':
    main()

