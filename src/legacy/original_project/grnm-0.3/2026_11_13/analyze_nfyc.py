import argparse
import ast
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from openpyxl import load_workbook


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = "SMESG000077821"


def read_table(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def sheet(path, index=0):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = book.worksheets[index].values
        header = next(rows)
        return [dict(zip(header, row)) for row in rows]
    finally:
        book.close()


def write_table(out, name, fields, rows):
    with (out / (name + ".tsv")).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields.split())
        writer.writerows(rows)


def neighbors(graph, tf):
    direct = graph.get(tf, set()) - {tf}
    paths = defaultdict(set)
    for intermediate in direct:
        for target in graph.get(intermediate, ()):
            if target != tf and target not in direct:
                paths[target].add(intermediate)
    return direct, paths


def joined(values):
    return ";".join(sorted(values))


def bh(values):
    result = [0.0] * len(values)
    running = 1.0
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    for rank, (index, p) in reversed(list(enumerate(ranked, 1))):
        running = min(running, p * len(values) / rank)
        result[index] = running
    return result


def run_null(graph, universe, responses, count, seed, candidates=None):
    rng = random.Random(seed)
    # Keep observed self-loops fixed and exclude creation of new loops.
    edges = sorted((u, v) for u, targets in graph.items() for v in targets if u != v)
    original = Counter(edges)
    state = {u: set(targets) for u, targets in graph.items()}
    if len(edges) < 2:
        raise ValueError("At least two non-loop edges are required")
    accepted = attempted = 0

    def swap(attempts):
        nonlocal accepted, attempted
        attempted += attempts
        for _ in range(attempts):
            i = rng.randrange(len(edges))
            j = rng.randrange(len(edges))
            u, v = edges[i]
            x, y = edges[j]
            if u == x or v == y or u == y or x == v or y in state[u] or v in state[x]:
                continue
            if candidates is not None and ((u, y) not in candidates or (x, v) not in candidates):
                continue
            state[u].remove(v)
            state[x].remove(y)
            state[u].add(y)
            state[x].add(v)
            edges[i], edges[j] = (u, y), (x, v)
            accepted += 1

    def overlaps(network):
        direct, indirect = neighbors(network, ROOT)
        return [
            len(set(level) & universe & response)
            for level in (direct, indirect) for response in responses.values()
        ]

    observed = overlaps(graph)
    swap(10 * len(edges))
    interval = max(1, len(edges) // 10)
    draws = []
    for _ in range(count):
        swap(interval)
        draws.append(overlaps(state))
    if not accepted:
        raise ValueError("No swaps accepted. Check the network and candidate restrictions")
    assert Counter(u for u, _ in original) == Counter(u for u, _ in edges)
    assert Counter(v for _, v in original) == Counter(v for _, v in edges)
    assert set(edges) == {(u, v) for u, targets in state.items() for v in targets if u != v}
    columns = list(zip(*draws))
    pvals = [(1 + sum(x >= obs for x in col)) / (count + 1) for obs, col in zip(observed, columns)]
    results = []
    labels = [(level, group) for level in (1, 2) for group in responses]
    for (level, group), obs, col, p, q in zip(labels, observed, columns, pvals, bh(pvals)):
        expected = mean(col)
        results.append((level, group, obs, expected, obs / expected if expected else None, p, q))
    return results, draws, labels, (attempted, accepted, interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--rna", type=Path, help="Complete tested RNAi table (TSV); columns gene_id, log2fc, padj")
    parser.add_argument("--rna-fdr", type=float)
    parser.add_argument("--min-abs-log2fc", type=float)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--candidates", type=Path, help="Optional TSV of allowed source,target pairs in canonical SMESG IDs")
    args = parser.parse_args()
    if args.rna and (
        args.rna_fdr is None or not 0 < args.rna_fdr <= 1
        or args.min_abs_log2fc is None or args.min_abs_log2fc < 0 or args.permutations < 1
    ):
        parser.error("--rna requires valid --rna-fdr, --min-abs-log2fc and a positive --permutations")
    args.out.mkdir(parents=True, exist_ok=True)
    paper = PROJECT / "paper_data"
    tf_rows = sheet(next(paper.glob("Transcription factors*.xlsx")))
    model = sheet(next(paper.glob("Predicted transcription factor*.xlsx")))
    aliases = defaultdict(set)
    for row in read_table(PROJECT / "map_data/adata_var_data.csv"):
        for field in ("gene_symbol", "gene_symbol_all_unique"):
            if row[field] and row[field].lower() != "nan":
                aliases[row[field]].add(row["smesg"])
    tf_ids = defaultdict(set)
    for row in tf_rows:
        tf_ids[row["TF_name"]].add(row["smesg_ID"])
    for tf, ids in tf_ids.items():
        aliases[tf].update(ids)

    graph = defaultdict(set)
    records = defaultdict(list)
    exclusions = Counter()
    audit = []
    for row in model:
        tf, target = row["Transcription factors"], row["target"]
        if not tf or not target:
            exclusions["missing_endpoint"] += 1
            continue
        if len(tf_ids[tf]) != 1:
            raise ValueError(f"Unresolved TF {tf}: {tf_ids[tf]}")
        source = next(iter(tf_ids[tf]))
        try:
            q = float(row["padj"])
        except (TypeError, ValueError):
            q = math.nan
        records[source, target].append((tf, row["estimate"], row["regions"], row["pval"], row["padj"]))
        if not 0 <= q < 0.05:
            exclusions["invalid_or_non_significant_padj"] += 1
            audit.append((source, target, tf, row["padj"]))
            continue
        graph[source].add(target)

    write_table(args.out, "network_excluded_rows", "source target tf padj", audit)
    direct, indirect = neighbors(graph, ROOT)
    incoming1 = defaultdict(set)
    incoming2 = defaultdict(set)
    for tf in graph:
        first, second = neighbors(graph, tf)
        for gene in first:
            incoming1[gene].add(tf)
        for gene in second:
            incoming2[gene].add(tf)

    figure = {}
    shown_edges = set()
    for sign in ("pos", "neg"):
        for row in read_table(PROJECT / f"example_data/NFYC_{sign}_edge.csv"):
            shown_edges.add((row["from"], row["to"], sign))
            for label in (row["from"], row["to"]):
                ids = {label} if label.startswith("SMESG") else aliases[label]
                figure[label] = ids
    universe_all = set(graph).union(*(graph.values()))
    figure_rows = []
    resolved = {label: next(iter(ids)) for label, ids in figure.items() if len(ids) == 1}
    resolved.update({"NFYC": ROOT, "IDH1": "SMESG000035181"})
    # Recover ID evidence from the default legacy tree inputs and the displayed parent/sign.
    legacy_candidates = defaultdict(set)
    for sign in ("pos", "neg"):
        legacy_path = PROJECT / f"example_data/edge_{sign}_1.csv"
        for line in legacy_path.read_text(encoding="utf-8-sig").splitlines():
            parent, targets = line.split(",", 1)
            for raw in ast.literal_eval(targets):
                if raw.startswith("SMESG"):
                    legacy_candidates[parent, sign].add(raw)
    recovered = set()
    for label, ids in figure.items():
        if label in resolved or not ids:
            continue
        supported = set()
        for parent, child, sign in shown_edges:
            if child == label:
                supported.update(ids & legacy_candidates[parent, sign])
        if len(supported) == 1:
            resolved[label] = next(iter(supported))
            recovered.add(label)
    write_table(args.out, "figure_edges", "source_label target_label sign source_id target_id in_fdr_graph", (
        (u, v, sign, resolved.get(u), resolved.get(v), int(resolved[v] in graph.get(resolved[u], ()))
         if u in resolved and v in resolved else None)
        for u, v, sign in sorted(shown_edges)
    ))
    evidence_pairs = set()
    summary = defaultdict(set)
    for label, ids in sorted(figure.items()):
        if label == "NFYC":
            continue
        gene = resolved.get(label)
        level = 1 if gene in direct else 2 if gene in indirect else None
        mids = indirect.get(gene, set())
        others1 = incoming1[gene] - {ROOT}
        others2 = incoming2[gene] - {ROOT}
        if level:
            summary[level, "total"].add(gene)
            if others1 or others2:
                summary[level, "shared"].add(gene)
        if gene:
            evidence_pairs.add((ROOT, gene))
            for mid in mids:
                evidence_pairs.update(((ROOT, mid), (mid, gene)))
        status = ("legacy_manual_id" if label == "IDH1" else "legacy_tree_match" if label in recovered else "resolved" if gene else
                  "ambiguous" if ids else "unmapped")
        figure_rows.append((
            label, gene, joined(ids), status, int(gene in universe_all), level, joined(mids),
            len(others1) if gene in universe_all else None, joined(others1),
            len(others2) if gene in universe_all else None, joined(others2),
            len(records.get((ROOT, gene), ())) if gene else None,
        ))
    write_table(args.out, "figure_gene_connectivity",
                "label gene_id candidate_ids mapping in_graph nfyc_distance intermediates other_tf_1_n other_tf_1 other_tf_2_n other_tf_2 direct_model_rows",
                figure_rows)
    write_table(args.out, "figure_model_evidence", "source_id target_id tf estimate regions pval padj", (
        (u, v, *record) for u, v in sorted(evidence_pairs) for record in records.get((u, v), ())
    ))
    write_table(args.out, "all_nfyc_neighbors", "gene_id nfyc_distance intermediates other_tf_1_n other_tf_2_n", (
        (gene, 1 if gene in direct else 2, joined(indirect.get(gene, ())),
         len(incoming1[gene] - {ROOT}), len(incoming2[gene] - {ROOT}))
        for gene in sorted(direct | set(indirect))
    ))
    old = read_table(PROJECT / "NFYC_edge_analysis/NFYC_all_edges.csv")
    old_exclusive = [row for row in old if row["NFYC_only"].lower() == "true"]
    audit_lines = [
        f"Model rows {len(model)}; exclusions {dict(exclusions)}.",
        f"Directed edges {sum(map(len, graph.values()))}; genes {len(universe_all)}; TF sources {len(graph)}.",
        f"Full NFYC graph neighborhoods: first {len(direct)}, second {len(indirect)}.",
        f"Legacy counts: {len(old_exclusive)}/{len(old)} exclusive rows; "
        f"{len({r['target'] for r in old_exclusive})}/{len({r['target'] for r in old})} exclusive unique targets.",
        f"Figure target labels: {len(figure_rows)}; unresolved {sum(row[1] is None for row in figure_rows)}; "
        f"resolved without an NFYC path within two steps {sum(row[1] is not None and row[5] is None for row in figure_rows)}.",
    ]
    for level in (1, 2):
        audit_lines.append(f"Figure order {level}: {len(summary[level, 'shared'])}/{len(summary[level, 'total'])} shared with other TFs within two steps.")
    for level, genes in ((1, direct), (2, indirect)):
        shared = sum(bool((incoming1[g] | incoming2[g]) - {ROOT}) for g in genes)
        audit_lines.append(f"Full graph order {level}: {shared}/{len(genes)} shared with other TFs within two steps.")
    if args.rna:
        tested = {}
        seen = set()
        for row in read_table(args.rna, "\t"):
            gene = row["gene_id"].strip()
            if gene in seen:
                raise ValueError(f"Duplicate RNAi gene: {gene}")
            seen.add(gene)
            p = row["padj"].strip()
            fc = row["log2fc"].strip()
            if p.lower() in ("", "na", "nan") or fc.lower() in ("", "na", "nan"):
                continue
            q, effect = float(p), float(fc)
            if not 0 <= q <= 1 or not math.isfinite(effect):
                raise ValueError(f"Invalid RNAi statistics for {gene}")
            tested[gene] = q, effect
        universe = universe_all & tested.keys()
        if not universe:
            raise ValueError("No tested RNAi genes match the itGRN IDs")
        de = {g for g in universe if tested[g][0] < args.rna_fdr and abs(tested[g][1]) >= args.min_abs_log2fc}
        responses = {"all": de, "up": {g for g in de if tested[g][1] > 0}, "down": {g for g in de if tested[g][1] < 0}}
        candidates = None
        if args.candidates:
            candidates = {(r["source"], r["target"]) for r in read_table(args.candidates, "\t")}
            if any((u, v) not in candidates for u, ts in graph.items() for v in ts if u != v):
                raise ValueError("Candidate table must include every observed non-loop edge")
        null, draws, labels, diagnostics = run_null(graph, universe, responses, args.permutations, args.seed, candidates)
        write_table(args.out, "rna_null", "order response observed null_mean fold_enrichment empirical_p bh_q", null)
        write_table(args.out, "rna_null_draws", " ".join(f"order{d}_{g}" for d, g in labels), draws)
        matched = []
        for level in (1, 2):
            focal = (direct, indirect)[level - 1]
            size = len(set(focal) & universe)
            tolerance = max(1, math.ceil(0.1 * size))
            for group, response in responses.items():
                if not response or not size:
                    continue
                scores = []
                for tf in graph:
                    eligible = set(neighbors(graph, tf)[level - 1]) & universe
                    if eligible and (tf == ROOT or abs(len(eligible) - size) <= tolerance):
                        overlap = len(eligible & response)
                        fold = overlap * len(universe) / (len(eligible) * len(response))
                        scores.append((tf, len(eligible), overlap, fold))
                for tf, n, hits, fold in sorted(scores, key=lambda r: (-r[3], r[0])):
                    rank = 1 + sum(other[3] > fold for other in scores)
                    matched.append((level, group, tf, n, hits, fold, rank, len(scores), tolerance))
        write_table(args.out, "rna_matched_tf", "order response tf_id tested_neighbors overlap fold_enrichment rank n_tf size_tolerance", matched)
        audit_lines.extend((
            f"RNAi: {args.rna}; FDR < {args.rna_fdr}; abs(log2FC) >= {args.min_abs_log2fc}.",
            f"Eligible tested genes: {len(universe)}; responses: { {k: len(v) for k,v in responses.items()} }.",
            f"Candidates: {args.candidates}; seed {args.seed}; attempted, accepted, interval: {diagnostics}.",
        ))
    else:
        audit_lines.append("RNAi enrichment and matched TF comparison pending a complete, independently tested planarian Nfyc RNAi table.")
    (args.out / "summary.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

