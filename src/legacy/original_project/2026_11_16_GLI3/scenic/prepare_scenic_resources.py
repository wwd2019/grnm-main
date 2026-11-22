from pathlib import Path
import csv

here = Path(__file__).resolve().parent
resources = here/"resources"
source = resources/"motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
with source.open(encoding="utf-8") as handle, (resources/"scenic_motif_annotation.tsv").open("w",encoding="utf-8",newline="") as out:
    writer = csv.writer(out,delimiter="\t")
    writer.writerow(("motif","TF","annotationSource"))
    for row in csv.DictReader(handle,delimiter="\t"):
        if not row["gene_name"]:
            continue
        ortholog = row["orthologous_identity"] not in ("1","1.0")
        similar = float(row["motif_similarity_qvalue"]) > 0
        label = {(False,False):"directAnnotation",(True,False):"inferredBy_Orthology",
                 (False,True):"inferredBy_MotifSimilarity",(True,True):"inferredBy_MotifSimilarity_n_Orthology"}[ortholog,similar]
        writer.writerow((row["#motif_id"],row["gene_name"],label))
vendor = next((here/"vendor").glob("aertslab-SCENIC-*"))/"R"
path = vendor/"class_ScenicOptions.R"
code = path.read_text(encoding="utf-8")
code = code.replace("dbIndexCol='features')","dbIndexCol='features', dbAnnotFiles=NULL)",1)
code = code.replace("db_annotFiles=NULL,","db_annotFiles=dbAnnotFiles,",1)
path.write_text(code,encoding="utf-8")
path = vendor/"runSCENIC_2_createRegulons.R"
code = path.read_text(encoding="utf-8").replace("columns=allGenes)","columns=allGenes, indexCol=dbIndexCol)")
path.write_text(code,encoding="utf-8")
