"""
grn_common.py — Phase 2/3/4 共享的基因身份工具
================================================================
集中三件事, 避免在 phase2/3/4 各自重复实现导致身份不一致:

  1. DEFAULT_WATCHLIST : 重点展示基因列表 (单一来源)
  2. 路径常量          : ROOT / MAP_DIR / 各 phase 输出目录
  3. load_id_maps()    : SMED / SMESG / symbol 三向映射 (含版本后缀回退)
  4. resolve_watchlist(): 把重点基因解析成 canonical 身份 + 状态

状态含义 (互斥, 按优先级取第一个命中):
  rna_missing   : symbol 在映射表里根本没有对应 SMED/SMESG  (HES4/DMRTA1/CREB5)
  unmapped      : 有 symbol 但既不在 z 空间也无可解析 ID
  predicted     : 拥有 phase2 z 向量, 可作为 phase3 的 target
  eligible_source: 拥有 phase2 z 向量且可作为 phase3 的 source(TF 侧)
"""

import os

# ── 路径 ──
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
NEW_DATA_DIR = os.path.join(ROOT_DIR, 'data')
MAP_DIR = os.path.join(ROOT_DIR, 'map_data')
RAW_DIR = os.path.join(ROOT_DIR, 'raw_data')
PHASE1_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase1_output')
PHASE2_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase2_output')
PHASE3_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase3_output')

# ── 单一重点基因列表 ──
# 涡虫 (Schmidtea mediterranea) 没有人类 HES4/DMRTA1/CREB5 的 1:1 直系同源
# (NCBI/Ensembl/UniProt/OrthoDB 均查不到), 只有家族成员。
# 因此 watchlist 使用涡虫基因组中真实存在、且在 GLI3 GRN 中代表同源家族的基因:
#   HES4  → Hes1    (SMED30006440 / SMESG000053680, 人 HES4 的涡虫直系同源, hesl-1)
#   DMRTA1→ DMRT1   (SMED30019616 / SMESG000002896, DMRT 家族)
#   CREB5 → CREB3L1 (SMED30015546 / SMESG000059546, CREB/ATF 家族)
# 人源对照 (已验证 Ensembl REST):
#   HES4=ENSG00000188290  DMRTA1=ENSG00000176399  CREB5=ENSG00000146592
DEFAULT_WATCHLIST = ['Hes1', 'DMRT1', 'CREB3L1', 'PAX6', 'HES5']


def load_id_maps(root=ROOT_DIR):
    """读取 adata_var_data.csv, 返回三向映射 + 版本后缀回退表。

    返回 dict:
      smed2sym   : SMED -> gene_symbol (含 SMED 占位符名)
      smesg2sym  : SMESG -> gene_symbol (仅真实 symbol, 排除 SMED/SMESG 占位)
      sym2smed   : gene_symbol -> [SMED...] (一个 symbol 可能多个 SMED, 如 PAX6)
      smed2smesg : SMED -> SMESG
      base2sym   : SMESG base(去 -N 转录本后缀) -> gene_symbol
    """
    import pandas as pd
    ad = pd.read_csv(os.path.join(root, 'map_data', 'adata_var_data.csv'))
    sym = ad['gene_symbol'].astype(str)

    smed2sym = {str(k): v for k, v in zip(ad['smed'].astype(str), sym)}
    smesg2sym = {str(k): v for k, v in zip(ad['smesg'].astype(str), sym)
                 if v and not v.startswith(('SMED', 'SMESG'))}

    sym2smed = {}
    for smed, s in smed2sym.items():
        if s and not s.startswith(('SMED', 'SMESG')):
            sym2smed.setdefault(s, []).append(smed)

    smed2smesg = {str(k): str(v) for k, v in
                  zip(ad['smed'].astype(str), ad['smesg'].astype(str))}

    # SMESG 转录本后缀回退: SMESG000036216-1 / .1 -> base
    import re
    base2sym = {}
    for k, v in smesg2sym.items():
        m = re.match(r'^(SMESG\d+)[-]\d+$', k)
        if m:
            base2sym.setdefault(m.group(1), v)

    return dict(smed2sym=smed2sym, smesg2sym=smesg2sym, sym2smed=sym2smed,
                smed2smesg=smed2smesg, base2sym=base2sym)


def map_symbol(sym_id, maps, kind='smed'):
    """把一个 ID 映射到 symbol; kind='smesg' 时带版本后缀回退。未命中返回原 ID。"""
    if kind == 'smesg':
        import re
        v = maps['smesg2sym'].get(sym_id)
        if v:
            return v
        m = re.match(r'^(SMESG\d+)[-.]\d+$', str(sym_id))
        if m:
            return maps['base2sym'].get(m.group(1), sym_id)
        return sym_id
    return maps['smed2sym'].get(sym_id, sym_id)


def resolve_watchlist(watchlist, maps, z_smed_set):
    """把重点基因解析成 (gene -> dict) 身份状态表。

    z_smed_set: 拥有 phase2 z 的 SMED 集合 (all_gene_z 的 gene_ids)。

    每个 gene 返回:
      symbol, smeds, smesgs, has_id, in_z, status
    """
    out = {}
    for g in dict.fromkeys(watchlist):  # 去重保序
        smeds = maps['sym2smed'].get(g, [])
        smesgs = [maps['smed2smesg'].get(s) for s in smeds]
        smesgs = [x for x in smesgs if x]
        has_id = bool(smeds)
        in_z = [s for s in smeds if s in z_smed_set]
        if not has_id:
            status = 'rna_missing'
        elif not in_z:
            status = 'unmapped'
        else:
            status = 'predicted'
        out[g] = dict(symbol=g, smeds=smeds, smesgs=smesgs,
                      has_id=has_id, in_z=sorted(in_z), status=status)
    return out
