import pandas as pd
from pathlib import Path

# ============================================================
# 参数
# ============================================================

# 修改成你的 xlsx 文件名
INPUT_FILE = "./paper_data/Predicted transcription factor–target relationships in the planarian inter-tissue gene regulatory network.xlsx"

# 要查询的 Transcription factor
TF_NAME = "NFYC"

# 输出目录
OUTPUT_DIR = Path("NFYC_edge_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. 读取 Excel
# ============================================================

print("=" * 80)
print("读取 Excel 文件")
print("=" * 80)

df = pd.read_excel(INPUT_FILE)

print(f"文件：{INPUT_FILE}")
print(f"数据行数：{len(df)}")
print("\n列名：")
print(df.columns.tolist())


# ============================================================
# 2. 检查必要列
# ============================================================

required_columns = [
    "target",
    "Transcription factors",
    "estimate",
    "regions"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Excel 中缺少列：{col}\n"
            f"当前列名为：{df.columns.tolist()}"
        )


# ============================================================
# 3. 数据清洗
# ============================================================

df["target"] = (
    df["target"]
    .astype(str)
    .str.strip()
)

df["Transcription factors"] = (
    df["Transcription factors"]
    .astype(str)
    .str.strip()
)

df["regions"] = (
    df["regions"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. 找到所有 NFYC 边
# ============================================================

nfyc_df = df[
    df["Transcription factors"].str.upper()
    == TF_NAME.upper()
].copy()

print("\n" + "=" * 80)
print(f"{TF_NAME} 边统计")
print("=" * 80)

print(f"NFYC 总边数：{len(nfyc_df)}")


# ============================================================
# 5. 判断每条 NFYC 边是否为“仅 NFYC”
# ============================================================
#
# 判断单位：
#     target + regions
#
# 如果同一个 target + region 下：
#
#     NFYC
#
# 而没有：
#
#     NFYA / NFYB / 其他 TF
#
# 则认为这个边仅由 NFYC 对应。
# ============================================================

# 对每一个 target + region，
# 统计存在的所有 TF

group_tf = (
    df
    .groupby(
        ["target", "regions"],
        dropna=False
    )["Transcription factors"]
    .agg(
        lambda x: sorted(
            set(
                str(v).strip()
                for v in x
                if str(v).strip() != ""
            )
        )
    )
    .reset_index()
)

# TF 数量
group_tf["TF_count"] = (
    group_tf["Transcription factors"]
    .apply(len)
)

# 是否只有 NFYC
group_tf["NFYC_only"] = (
    group_tf["TF_count"].eq(1)
    &
    group_tf["Transcription factors"].apply(
        lambda x: len(x) == 1
        and x[0].upper() == TF_NAME.upper()
    )
)


# ============================================================
# 6. 将“仅 NFYC”信息合并回 NFYC 边
# ============================================================

nfyc_df = nfyc_df.merge(
    group_tf[
        [
            "target",
            "regions",
            "Transcription factors",
            "TF_count",
            "NFYC_only"
        ]
    ],
    on=["target", "regions"],
    how="left",
    suffixes=("", "_all")
)


# ============================================================
# 7. 筛选仅 NFYC 的边
# ============================================================

nfyc_only_df = nfyc_df[
    nfyc_df["NFYC_only"]
].copy()

print(
    f"仅 NFYC 的边数：{len(nfyc_only_df)}"
)

print(
    f"非仅 NFYC 的边数："
    f"{len(nfyc_df) - len(nfyc_only_df)}"
)


# ============================================================
# 8. 输出所有 NFYC 边
# ============================================================

all_nfyc_file = (
    OUTPUT_DIR /
    "NFYC_all_edges.xlsx"
)

nfyc_df.to_excel(
    all_nfyc_file,
    index=False
)


# ============================================================
# 9. 输出仅 NFYC 的边
# ============================================================

nfyc_only_file = (
    OUTPUT_DIR /
    "NFYC_only_edges.xlsx"
)

nfyc_only_df.to_excel(
    nfyc_only_file,
    index=False
)


# ============================================================
# 10. 输出 CSV 版本
# ============================================================

nfyc_df.to_csv(
    OUTPUT_DIR / "NFYC_all_edges.csv",
    index=False,
    encoding="utf-8-sig"
)

nfyc_only_df.to_csv(
    OUTPUT_DIR / "NFYC_only_edges.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 输出详细结果
# ============================================================

print("\n" + "=" * 80)
print("所有 NFYC 边")
print("=" * 80)

show_columns = [
    "target",
    "Transcription factors",
    "estimate",
    "regions",
    "TF_count",
    "NFYC_only"
]

print(
    nfyc_df[show_columns]
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("仅 NFYC 的边")
print("=" * 80)

print(
    nfyc_only_df[show_columns]
    .to_string(index=False)
)


# ============================================================
# 12. 汇总统计
# ============================================================

summary = pd.DataFrame({
    "TF": [TF_NAME],
    "total_edges": [len(nfyc_df)],
    "NFYC_only_edges": [len(nfyc_only_df)],
    "non_NFYC_only_edges": [
        len(nfyc_df) - len(nfyc_only_df)
    ],
    "NFYC_only_ratio": [
        len(nfyc_only_df) / len(nfyc_df)
        if len(nfyc_df) > 0 else 0
    ]
})

summary_file = (
    OUTPUT_DIR /
    "NFYC_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 最终结果
# ============================================================

print("\n" + "=" * 80)
print("最终统计")
print("=" * 80)

print(f"Transcription factors = {TF_NAME}")
print(f"总边数：{len(nfyc_df)}")
print(f"仅 {TF_NAME} 的边数：{len(nfyc_only_df)}")
print(
    f"其他 TF 共同存在的边数："
    f"{len(nfyc_df) - len(nfyc_only_df)}"
)

if len(nfyc_df) > 0:
    ratio = len(nfyc_only_df) / len(nfyc_df)
    print(f"仅 {TF_NAME} 比例：{ratio:.2%}")

print("\n输出文件：")
print(all_nfyc_file)
print(nfyc_only_file)
print(summary_file)