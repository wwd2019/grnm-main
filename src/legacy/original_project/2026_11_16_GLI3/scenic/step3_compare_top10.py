from pathlib import Path

import pandas as pd
import sys

here = Path(__file__).resolve().parent
audit = pd.read_csv(here.parent / "inputs/target_scores.tsv", sep="\t").set_index("target")
run = Path(sys.argv[1])
scores = pd.read_csv(run / "gli3_scenic.tsv", sep="\t").rename(columns={"gene":"target","CoexWeight":"importance"})
scores = scores[scores.target.isin(audit.index) & scores.importance.gt(0)]
top = scores.sort_values(["importance", "target"], ascending=[False, True]).drop_duplicates("target").head(10).copy()
top["rank"] = range(1,len(top)+1)
top["hit"] = top.target.map(audit.benchmark_positive)
top.to_csv(run / "scenic_top10.tsv", sep="\t", index=False)
summary = pd.read_csv(here.parent / "results/summary.tsv", sep="\t")
summary.loc[len(summary)] = ["R SCENIC (GENIE3 + RcisTarget)",len(scores),len(top),
                            int(top.hit.sum()),pd.NA,";".join(top.loc[top.hit,"target"])]
summary.to_csv(run / "comparison.tsv",sep="\t",index=False)
print(summary.to_string(index=False))

