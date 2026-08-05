"""
计算因子同质化率 (Jaccard Homogeneity)
对每组 registry 计算因子两两特征集 Jaccard 相似度，统计 >0.7 的占比
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.path.join(BASE, '..', '..', '..', 'outputs', 'legacy', 'root_factor_output_academic_v81')

# 各实验组的 registry 路径（指向 outputs/legacy/ 归档）
registries = {
    '完整模型':   os.path.join(LEGACY_DIR, 'run_20260720_171219', 'registry_academic.json'),
    '无MLQC':     os.path.join(LEGACY_DIR, 'ablation6_no_mlqc_20260721_145755', 'registry_academic.json'),
    '无GFlowNet': os.path.join(LEGACY_DIR, 'ablation8_no_gflownet_20260721_212241', 'registry_academic.json'),
    '无约束':     os.path.join(LEGACY_DIR, 'ablation5_no_constraints_20260721_200333', 'registry_academic.json'),
}

THRESHOLD = 0.7


def calc_jaccard_homogeneity(registry_path, label=""):
    if not os.path.exists(registry_path):
        print(f"⚠️ [{label}] 文件不存在: {registry_path}")
        return None

    with open(registry_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    factors = data.get('factors', {})
    factor_names = list(factors.keys())
    n_factors = len(factor_names)

    if n_factors < 2:
        print(f"⚠️ [{label}] 因子数不足 (n={n_factors})，跳过")
        return 0.0

    feat_sets = [set(factors[fid]['features']) for fid in factor_names]

    high_sim_pairs = 0
    total_pairs = 0

    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            set1, set2 = feat_sets[i], feat_sets[j]
            if not set1 or not set2:
                continue
            jaccard = len(set1 & set2) / len(set1 | set2)
            total_pairs += 1
            if jaccard > THRESHOLD:
                high_sim_pairs += 1

    rate = high_sim_pairs / total_pairs if total_pairs > 0 else 0.0
    print(f"✅ [{label}] 因子数: {n_factors} | 总因子对: {total_pairs} | "
          f"Jaccard>{THRESHOLD} 对数: {high_sim_pairs} | 同质化率: {rate:.2%}")
    return rate


if __name__ == '__main__':
    print("=" * 60)
    print("因子结构同质化率 (Jaccard Homogeneity) 计算")
    print(f"阈值: Jaccard > {THRESHOLD} 视为高度相似")
    print("=" * 60)

    results = {}
    for label, path in registries.items():
        r = calc_jaccard_homogeneity(path, label)
        if r is not None:
            results[label] = r

    print("\n" + "=" * 60)
    print("汇总:")
    print("-" * 60)
    for label, rate in results.items():
        bar_len = int(rate * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {label:12s} | {rate:.2%} {bar}")
    print("=" * 60)