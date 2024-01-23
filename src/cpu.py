import numpy as np
from sklearn.cluster import KMeans
import scanpy as sc
import sklearn.covariance as skc
import anndata
import pandas as pd

class cicero:
    '''
    parameters:
        binmat:list or np.ndarray or np.matrix or scipy sparse matrix or AnnData or pandas dataframe
        binary accessibility values matrix A, where Amn is zero if no read was observed to overlap peak m in cell n and one otherwise.
        
        peakinfo:pandas dataframe or list or None, default=None
        the bed file of peaks. If binmat is anndata or dataframe with peakinfo, this parameter is not needed.
        
        reduce:str,{'tsne','umap'},default='tsne'
        method of mapping cells into low dimensions
    '''
    def __init__(self,binmat,peakinfo=None,reduce='tsne') -> None:
        
        if isinstance(binmat,anndata.AnnData):
            self.adata=binmat
        else:
            self.adata=anndata.AnnData(X=binmat)
        sc.tl.pca(self.adata, svd_solver='arpack')
        sc.pp.neighbors(self.adata, n_neighbors=50)
        if reduce=='tsne' or 'TSNE' or 'tSNE':
            sc.tl.tsne(self.adata,n_components=2)
        elif reduce=='umap' or 'UMAP':
            sc.tl.umap(self.adata,n_components=2)
        else:
            raise ValueError('reduce must be tsne or umap')
        self.umap=self.adata.obsm['X_umap']
        self.cluster=KMeans(n_clusters=self.umap.shape[0]//50,random_state=0).fit_predict(self.umap)
        group = []
        for i in range(0,self.cluster.max()+1):
            indices = np.where(self.cluster == i)
            group.append(np.sum(self.adata.X[indices],axis=0).A[0])
        self.group=anndata.AnnData(X=np.array(group))
        sc.pp.normalize_total(self.group, target_sum=1e4)
        print('init done')