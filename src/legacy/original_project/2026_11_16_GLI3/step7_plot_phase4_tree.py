from pathlib import Path
import argparse
import subprocess

import pandas as pd

here = Path(__file__).resolve().parent
results = here / "results"
parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
args = parser.parse_args()

final = pd.read_csv(results / "final_network.tsv", sep="\t")
pando = pd.read_csv(here / "inputs/TF_gene_network_us.csv")
direct = final.assign(tf="GLI3", estimate=final.weight)
original_sign = (pando[pando.tf.eq("GLI3")].sort_values("padj_min")
                 .drop_duplicates("target").set_index("target").estimate)
seed = direct.stage.eq("Pando input")
direct.loc[seed, "estimate"] = direct.loc[seed, "target"].map(original_sign)
downstream = pando[pando.tf.isin(final.target)
                   & ~pando.target.isin(set(final.target) | {"GLI3"})
                   & pando.padj_min.between(0, .05)].copy()
downstream = (downstream.assign(magnitude=downstream.estimate.abs())
              .sort_values(["magnitude", "tf", "target"], ascending=[False, True, True])
              .drop_duplicates(["tf", "target"]).groupby("tf").head(5)
              .drop_duplicates("target"))
columns = ["tf", "target", "estimate"]
edges = pd.concat([direct[columns], downstream[columns]], ignore_index=True)
assert len(direct) == 29 and not downstream.empty and edges.estimate.notna().all()
edges.to_csv(results / "gli3_phase4_tree_edges.csv", index=False)
print(f"{len(direct)} direct edges; {len(downstream)} downstream edges via {downstream.tf.nunique()} TFs", flush=True)
hits = ",".join(final.loc[final.hit, "target"])
# Adapt the Phase 4 renderer locally to draw unique edges and emphasize direct hits.
renderer = (args.project / "draw_grn_tree.R").read_text(encoding="utf-8")
renderer = renderer.replace("edges <- bind_rows(layer1, layer2, layer3)",
    "edges <- bind_rows(layer1, layer2, layer3) %>% distinct(from, to, .keep_all = TRUE)\n"
    "edges$hit <- edges$from == root_gene & edges$to %in% highlight_genes")
old = '    geom_edge_diagonal(aes(colour = factor(attr)), alpha = 0.5, linewidth = 0.25) +'
assert old in renderer
renderer = renderer.replace(old, old + '\n    geom_edge_diagonal(aes(filter = hit, colour = factor(attr)), alpha = 1, linewidth = 1.05) +')
local_renderer = results / "phase4_tree_renderer.R"
local_renderer.write_text(renderer, encoding="utf-8")
for ext in ("png", "pdf"):
    output = results / f"gli3_phase4_tree.{ext}"
    before = output.stat().st_mtime_ns if output.exists() else None
    run = subprocess.run(["Rscript", str(local_renderer),
                          "--edges", str(results / "gli3_phase4_tree_edges.csv"),
                          "--root", "GLI3", "--output", str(output),
                          "--n1", str(len(direct)), "--nhubs", "6", "--n2", "5",
                          "--highlight", hits, "--spread-labels", "--label-all"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(run.stdout, flush=True)
    # This Windows R runtime can return an access-violation code at shutdown after ggsave.
    signature = b"%PDF-" if ext == "pdf" else b"\x89PNG\r\n\x1a\n"
    saved = (output.exists() and output.stat().st_mtime_ns != before
             and output.read_bytes().startswith(signature) and "Done" in run.stdout)
    if not saved or run.returncode not in (0, 3221225477):
        raise RuntimeError(run.stderr)

