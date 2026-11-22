from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent

def pando_rank(path, p_column, universe):
    edges = pd.read_csv(path)
    keep = edges.tf.eq("GLI3") & edges.target.isin(universe) & edges[p_column].between(0, 0.05)
    return (edges.loc[keep].assign(magnitude=lambda x: x.estimate.abs())
            .sort_values([p_column, "magnitude", "target"], ascending=[True, False, True])
            .drop_duplicates("target"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, default=HERE/"frozen/human_GLI3_predictions.csv")
    parser.add_argument("--out", type=Path, default=HERE/"results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = HERE/"inputs"
    old_audit = pd.read_csv(inputs/"target_scores.tsv", sep="\t")
    universe = set(old_audit.target)
    candidates = pd.read_csv(inputs/"PRIORITY_strong_DE_top10pct_CUTTag_proxy.tsv", sep="\t")
    positives = set(candidates.gene_symbol.dropna()) & universe
    assert positives == set(old_audit.loc[old_audit.benchmark_positive,"target"])

    source = inputs/"TF_gene_network_us.csv"
    first = pando_rank(source, "padj_min", universe)
    raw = pd.read_csv(source)
    motif = pd.read_csv(inputs/"motif2tf_human.tsv", sep="\t")
    input_targets = set(raw.loc[raw.tf.eq("GLI3"),"target"]) | set(motif.loc[motif.motif_tf.eq("GLI3"),"tf"])
    prediction = pd.read_csv(args.prediction)
    assert prediction.target.is_unique
    prediction = prediction[prediction.target.isin(universe)].copy()
    seed = prediction.target.isin(first.target)
    novel = ~prediction.target.isin(input_targets) & prediction.pred_class.ne(0) & prediction.weight.abs().ge(1.3)
    final = prediction.loc[seed | novel].assign(
        score=lambda x:x.prob.where(x.pred_class.ne(0),0),
        stage=lambda x:x.target.isin(first.target).map({True:"Pando input",False:"GRNM expansion"}),
    ).sort_values(["score","target"],ascending=[False,True])
    metacells = pando_rank(inputs/"metacells.csv", "padj", universe)
    tables = {
        "Our Pando": first.assign(score=first.padj_min, stage="Pando input"),
        "Our Pando + GRNM": final,
        "Pando metacells": metacells.assign(score=metacells.padj, stage="External Pando"),
    }
    summaries, details = [], []
    for name, edges in tables.items():
        top = edges.head(10)[["target","score","stage"]].copy()
        top.insert(0,"rank",range(1,len(top)+1))
        top.insert(0,"network",name)
        top["hit"] = top.target.isin(positives)
        summaries.append((name,len(edges),len(top),int(top.hit.sum()),
                          int((top.hit & top.stage.eq("GRNM expansion")).sum()),
                          ";".join(top.loc[top.hit,"target"])))
        details.append(top)
    summary = pd.DataFrame(summaries,columns=["network","total_edges","evaluated","hits","new_hits","genes"])
    summary.to_csv(args.out/"summary.tsv",sep="\t",index=False)
    pd.concat(details).to_csv(args.out/"top10_edges.tsv",sep="\t",index=False)
    final.assign(hit=final.target.isin(positives)).to_csv(args.out/"final_network.tsv",sep="\t",index=False)
    if args.prediction.resolve() == (HERE/"frozen/human_GLI3_predictions.csv").resolve():
        assert summary.total_edges.tolist() == [28,29,49]
        assert summary.hits.tolist() == [1,3,2]
        assert summary.new_hits.tolist() == [0,0,0]
    fig, ax = plt.subplots(figsize=(6.2,3.6), layout="constrained")
    bars = ax.bar(summary.network, summary.hits, color=["#577c91","#117765","#949da4"],width=.6)
    ax.bar_label(bars,labels=[f"{h}/{n}" for h,n in zip(summary.hits,summary.evaluated)],padding=4)
    ax.set(ylabel="GLI3 candidate hits",ylim=(0,4),yticks=range(5))
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(args.out/"top10.pdf")
    fig.savefig(args.out/"top10.png",dpi=220)
    plt.close(fig)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
