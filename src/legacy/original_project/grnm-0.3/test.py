import pandas as pd
from pathlib import Path

# ============================================================
# 参数
# ============================================================

EDGE_FILE = "./example_data/edges.csv"
ANNOTATION_FILE = "./raw_data/adata_var_data.csv"

# 输入需要查询的 SMESG
SOURCE_NODE = "SMESG000077821"

# 输出目录
OUTPUT_DIR = Path("edge_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. 读取 edgse.csv
# ============================================================

edge_df = pd.read_csv(EDGE_FILE)

if "source" not in edge_df.columns or "target" not in edge_df.columns:
    raise ValueError(
        "edgse.csv 必须包含 source 和 target 两列"
    )

edge_df["source"] = edge_df["source"].astype(str).str.strip()
edge_df["target"] = edge_df["target"].astype(str).str.strip()

print("=" * 80)
print("读取边数据")
print("=" * 80)
print(f"边数量：{len(edge_df)}")


# ============================================================
# 2. 读取基因注释
# ============================================================

anno_df = pd.read_csv(ANNOTATION_FILE)

# SMESG列
if "smesg" in anno_df.columns:
    smesg_col = "smesg"
else:
    smesg_col = anno_df.columns[0]

# 基因名列
gene_name_candidates = [
    "gene_symbol_all_unique",
    "gene_symbol_all",
    "gene_symbol",
    "uniprot_gn_symbol",
    "gene_symbol_pl",
    "smed"
]

gene_name_col = None

for col in gene_name_candidates:
    if col in anno_df.columns:
        gene_name_col = col
        break

if gene_name_col is None:
    raise ValueError(
        "adata_var_name.csv 中没有找到基因名列"
    )

anno_df[smesg_col] = (
    anno_df[smesg_col]
    .astype(str)
    .str.strip()
)

anno_df[gene_name_col] = (
    anno_df[gene_name_col]
    .astype(str)
    .str.strip()
)

# nan / 空值处理
anno_df.loc[
    anno_df[gene_name_col].isin(
        ["", "nan", "None", "NA", "N/A", "null"]
    ),
    gene_name_col
] = pd.NA


# ============================================================
# 3. 建立 SMESG → gene name 映射
# ============================================================

mapping_df = (
    anno_df[
        [smesg_col, gene_name_col]
    ]
    .dropna(subset=[smesg_col])
    .drop_duplicates(subset=[smesg_col])
)

gene_mapping = dict(
    zip(
        mapping_df[smesg_col],
        mapping_df[gene_name_col]
    )
)


def smesg_to_gene(smesg):
    """
    SMESG → gene name
    如果没有对应基因名，则保留 SMESG
    """
    gene = gene_mapping.get(smesg)

    if pd.notna(gene) and str(gene).strip() != "":
        return str(gene).strip()

    return smesg


# 查询的 source 对应基因名
source_gene = smesg_to_gene(SOURCE_NODE)

print("\n" + "=" * 80)
print("查询对象")
print("=" * 80)
print(f"SMESG ：{SOURCE_NODE}")
print(f"Gene  ：{source_gene}")


# ============================================================
# 4. 找到指定 source 的所有 target
# ============================================================

source_df = edge_df[
    edge_df["source"] == SOURCE_NODE
].copy()

if len(source_df) == 0:
    raise ValueError(
        f"edgse.csv 中没有找到 source = {SOURCE_NODE}"
    )

targets = (
    source_df["target"]
    .drop_duplicates()
    .tolist()
)

print(f"\n{SOURCE_NODE} 共指向 {len(targets)} 个 target")


# ============================================================
# 5. 检查每个 target 的所有 source
# ============================================================

results = []

for target in targets:

    # 所有指向这个 target 的 source
    incoming = edge_df[
        edge_df["target"] == target
    ]

    all_sources = (
        incoming["source"]
        .drop_duplicates()
        .tolist()
    )

    # 判断是否只有当前 source
    exclusive = (
        len(all_sources) == 1
        and all_sources[0] == SOURCE_NODE
    )

    # 获取当前边的 weight
    current_edge = source_df[
        source_df["target"] == target
    ]

    if "weight" in current_edge.columns:
        weight = current_edge.iloc[0]["weight"]
    else:
        weight = None

    # 所有 source 转成基因名
    all_source_genes = [
        smesg_to_gene(x)
        for x in all_sources
    ]

    # target 转成基因名
    target_gene = smesg_to_gene(target)

    results.append({
        # SMESG
        "source_smesg": SOURCE_NODE,
        "target_smesg": target,

        # 基因名
        "source_gene": source_gene,
        "target_gene": target_gene,

        # weight
        "weight": weight,

        # target 有多少个 source
        "number_of_sources": len(all_sources),

        # 所有 source
        "all_sources_smesg": ";".join(all_sources),
        "all_sources_gene": ";".join(all_source_genes),

        # 是否只有当前 source
        "exclusive_to_source": exclusive
    })


result_df = pd.DataFrame(results)


# ============================================================
# 6. 保存所有 target
# ============================================================

all_targets_file = (
    OUTPUT_DIR /
    f"{SOURCE_NODE}_all_targets.csv"
)

result_df.to_csv(
    all_targets_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. 筛选独占 target
# ============================================================

exclusive_df = result_df[
    result_df["exclusive_to_source"]
].copy()

exclusive_file = (
    OUTPUT_DIR /
    f"{SOURCE_NODE}_exclusive_targets.csv"
)

exclusive_df.to_csv(
    exclusive_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 输出结果
# ============================================================

print("\n" + "=" * 80)
print("所有 target")
print("=" * 80)

for _, row in result_df.iterrows():

    print(
        f"{row['target_smesg']:20s}"
        f" → {row['target_gene']:30s}"
        f" | source数量 = {row['number_of_sources']}"
    )


print("\n" + "=" * 80)
print("只以该 SMESG 为 source 的 target")
print("=" * 80)

print(f"共有 {len(exclusive_df)} 个\n")

for _, row in exclusive_df.iterrows():

    print(
        f"{row['target_smesg']:20s}"
        f" → {row['target_gene']}"
    )


# ============================================================
# 9. 保存汇总
# ============================================================

summary = pd.DataFrame({
    "source_smesg": [SOURCE_NODE],
    "source_gene": [source_gene],
    "total_targets": [len(result_df)],
    "exclusive_targets": [len(exclusive_df)],
    "non_exclusive_targets": [
        len(result_df) - len(exclusive_df)
    ]
})

summary_file = (
    OUTPUT_DIR /
    f"{SOURCE_NODE}_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 完成
# ============================================================

print("\n" + "=" * 80)
print("完成")
print("=" * 80)

print(f"查询 SMESG：{SOURCE_NODE}")
print(f"对应基因名：{source_gene}")
print(f"所有 target：{len(result_df)}")
print(f"独占 target：{len(exclusive_df)}")
print(
    f"非独占 target："
    f"{len(result_df) - len(exclusive_df)}"
)

print("\n输出文件：")
print(all_targets_file)
print(exclusive_file)
print(summary_file)