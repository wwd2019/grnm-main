"""Functions for reconstructing saved pando+GRN predictions.

This intentionally matches the archived pando+GRN training run.
The publication display name is pando+GRN; scores are not calibrated probabilities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
INPUTS=ROOT/'dataset/inputs'
SOURCE=ROOT/'dataset/scenic'
MODEL=ROOT/'results/model'

def reconstruct_gli3_scores():
    """Rebuild features without fitting, then score the fixed 1,794-gene universe."""
    audit=pd.read_csv(INPUTS/'target_scores.tsv',sep='\t')
    source=SOURCE
    allgenes=(source/'genes.txt').read_text().splitlines()
    names=sorted((set(audit.target)|{'GLI3','HES4'}) & set(allgenes))
    idx={g:i for i,g in enumerate(names)}; universe=set(names)-{'GLI3'}
    edges=pd.read_csv(INPUTS/'TF_gene_network_us.csv')
    edges=edges[edges.padj_min.between(0,.05)&edges.tf.isin(idx)&edges.target.isin(universe)&edges.tf.ne(edges.target)][['tf','target']].drop_duplicates()
    counts=edges[edges.tf.ne('GLI3')].groupby('tf').size()
    eligible=counts[counts>=10].index.to_numpy()
    training,held=train_test_split(eligible,test_size=.4,random_state=42)
    validation,test=train_test_split(held,test_size=.5,random_state=42)
    training=set(training)|set(counts[counts<10].index)
    assert not (training & set(validation) or training & set(test)) and 'GLI3' not in training
    train=edges[edges.tf.isin(training)]
    degree=np.bincount(train.target.map(idx).to_numpy(),minlength=len(names))
    columns={g:i for i,g in enumerate(allgenes)}
    expr=mmread(source/'expression.mtx').tocsr()[[columns[g] for g in names]].toarray().astype(np.float32)
    expr-=expr.mean(axis=1,keepdims=True)
    expr/=np.maximum(np.sqrt((expr*expr).sum(axis=1))[:,None],1e-12)
    with threadpool_limits(limits=2): corr=expr@expr.T
    boost=pd.read_csv(ROOT/'results/scenic/adjacencies.tsv',sep='\t')
    boost=boost[boost.TF.isin(idx)&boost.target.isin(idx)]
    B=np.zeros((len(names),len(names)),dtype=np.float32)
    B[boost.TF.map(idx),boost.target.map(idx)]=boost.importance.to_numpy(dtype=np.float32)
    motif=pd.read_csv(INPUTS/'motif2tf_human.tsv',sep='\t')
    motif=motif[motif.motif_tf.isin(idx)&motif.tf.isin(idx)]
    M=np.zeros((len(names),len(names)),dtype=np.float32)
    for r in motif.itertuples(): M[idx[r.motif_tf],idx[r.tf]]=max(M[idx[r.motif_tf],idx[r.tf]],float(r.prc_similarity))
    si=idx['GLI3']; tg=np.array([i for i,g in enumerate(names) if g!='GLI3'])
    x=np.column_stack([np.log1p(degree[tg]),np.full(len(tg),np.log1p(degree[si])),np.log1p(degree[tg]),np.log1p(B[si,tg]),corr[si,tg],M[si,tg]]).astype(np.float32)
    norm=np.load(MODEL/'normalization.npz'); weights=np.load(MODEL/'model_weights_numpy.npz')
    z=(x-norm['mean'])/norm['std']
    z=np.maximum(z@weights['0.weight'].T+weights['0.bias'],0)
    z=np.maximum(z@weights['3.weight'].T+weights['3.bias'],0)
    z=z@weights['5.weight'].T+weights['5.bias']
    score=(1/(1+np.exp(-z))).ravel()
    result=pd.DataFrame({'target':np.array(names)[tg],'score':score}).sort_values(['score','target'],ascending=[False,True]).reset_index(drop=True)
    return result
