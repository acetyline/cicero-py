import numpy as np
import sklearn.neighbors as skn
import sklearn.manifold as skm

class cicero:
    '''
    parameters:
        binmat:list or np.ndarray or np.matrix or scipy sparse matrix
        binary accessibility values matrix A, where Amn is zero if no read was observed to overlap peak m in cell n and one otherwise.
        
        reduce:str,{'tsne','umap'},default='tsne'
        method of mapping cells into low dimensions
    '''
    def __init__(self,binmat,reduce='tsne') -> None:
        if reduce=='tsne':
            self.reduce=skm.TSNE(n_components=2)
        elif reduce=='umap':
            self.reduce=skm.UMAP(n_components=2)
        else:
            raise ValueError('reduce must be one of {tsne,umap}')
        self.binmat=binmat
        
