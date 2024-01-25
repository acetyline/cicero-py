import numpy as np
from sklearn.cluster import KMeans
import scanpy as sc
import anndata
import pandas as pd
from scipy.optimize import minimize
from sklearn import preprocessing


class graphicalLasso:
    '''
        different from sklearn GraphicalLasso: rho is a matrix, not a integer
    '''
    def __init__(self, rho=None, maxItr=1e+3, tol=1e-2):
        self.rho = rho
        self.maxItr = int(maxItr)
        self.tol = tol
        self.scaler = None

    def fit(self, X):
        n_samples, n_features = X.shape[0], X.shape[1]

        self.scaler = preprocessing.StandardScaler().fit(X)
        self.X = self.scaler.transform(X)

        S = self.X.T.dot(self.X) / n_samples

        A = np.linalg.pinv(S)
        A_old = A
        invA = S

        for i in range(self.maxItr):
            for j in range(n_features):
                R, s, sii = self.__get(S)
                W = self.__get(invA)[0]
                L = self.__get(A)[0]

                sigma = sii + np.linalg.norm(self.rho)

                U, D, V = np.linalg.svd(W)
                W_half = U.dot(np.diag(np.sqrt(D)).dot(U.T))

                b = np.linalg.pinv(W_half).dot(s)

                beta = self.lasso(W_half, b, self.rho)

                w = W.dot(beta)

                l = -beta / (sigma - beta.T.dot(W).dot(beta))
                lmbd = 1 / (sigma - beta.T.dot(W).dot(beta))

                A = self.__put(L, l, lmbd)
                invA = self.__put(W, w, sigma)
                S = self.__put(R, s, sii)

            if np.linalg.norm(A - A_old, ord=2) < self.tol:
                break
            else:
                A_old = A

        self.covariance = S
        self.precision = A
        return self

    def __get(self, S):
        end = S.shape[0] - 1
        R = S[:-1, :-1]
        s = S[end, :-1]
        sii = S[end][end]

        return [R, s, sii]

    def __put(self, R, s, sii):
        n = R.shape[0] + 1
        X = np.empty([n, n])
        X[1:, 1:] = R
        X[1:, 0] = s
        X[0, 1:] = s
        X[0][0] = sii

        return X

    def lasso(self, W_half, b, rho):
        def objective(beta):
            return 0.5 * np.linalg.norm(W_half @ beta - b, 2) ** 2 + np.sum(np.abs(rho @ beta))

        beta_init = np.zeros(W_half.shape[1])

        res = minimize(objective, beta_init, method='SLSQP')
        
        return res.x
    
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
        umap=self.adata.obsm['X_umap']
        self.cluster=KMeans(n_clusters=umap.shape[0]//50,random_state=0).fit_predict(self.umap)
        group = []
        for i in range(0,self.cluster.max()+1):
            indices = np.where(self.cluster == i)
            group.append(np.sum(self.adata.X[indices],axis=0).A[0])
        self.group=anndata.AnnData(X=np.array(group))
        sc.pp.normalize_total(self.group, target_sum=1e4)
        print('init done')
    
    
    def runcicero(self,covmethod='graphical_lasso'):
        graphical=skc.GraphicalLasso()