import numpy as np
import os
import scipy.sparse as sp 
import scipy.io
import copy
from scipy import pi
import sys
import pandas as pd
from os.path import join
import gzip
import scanpy as sc
from scipy.io import mmwrite,mmread
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler


def check_symmetric(a, rtol=1e-03, atol=1e-03):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def calculate_binary_adj_matrix(coor, rad_cutoff=None, k_cutoff=None, model='Radius', return_indice=False):
    import sklearn.neighbors
    assert(model in ['Radius', 'KNN'])
    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
    A = np.empty((coor.shape[0], coor.shape[0]))
    for i in range(indices.shape[0]):
        for j in indices[i]:
            A[i][j] = 1
            A[j][i] = 1
    if return_indice:
        return A.astype(np.float32), indices
    else:
        return A.astype(np.float32)


def get_ground_truth_adj_matrix(coor, labels, n_neighbors=6, return_indice=False):
    import sklearn.neighbors
    uniq_labels = np.unique(labels)
    adj_indices = np.empty((coor.shape[0],n_neighbors)).astype(np.int16)
    A =  np.empty((coor.shape[0], coor.shape[0]))
    for each in uniq_labels:
        sample_idx = [i for i,item in enumerate(labels) if item==each]
        dic = dict(zip(np.arange(len(sample_idx)), sample_idx))
        coor_sub = coor[sample_idx,:]
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(coor_sub)
        distances, indices = nbrs.kneighbors(coor_sub)
        adj_indices[sample_idx,:] = np.vectorize(dic.get)(indices)
    for i in range(adj_indices.shape[0]):
        for j in adj_indices[i]:
            A[i][j] = 1
            A[j][i] = 1
    if return_indice:
        return A.astype(np.float32), adj_indices
    else:
        return A.astype(np.float32)

class Spatial_DLPFC_sampler(object):
    def __init__(self, slice_id='151673'):
        adata = sc.read_visium(path='/home/users/liuqiao/work/spatialRT/data/DLPFC/%s'%slice_id, count_file='%s_filtered_feature_bc_matrix.h5'%slice_id)
        adata.var_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var.highly_variable]
        print('highly varible genes', adata.shape)
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
        #load annotation label
        annot_df = pd.read_csv('/home/users/liuqiao/work/spatialRT/data/DLPFC/DLPFC_annotations/%s_truth.txt'%slice_id, sep='\t', header=None, index_col=0)
        #annot_df.fillna('NA', inplace=True)
        annot_df.columns = ['annotation']
        adata.obs['annotation'] = annot_df.loc[adata.obs_names, 'annotation']
        #filter NA cells/spots
        select_idx = pd.notnull(adata.obs['annotation'])    
        adata = adata[select_idx,:]
        print(adata.shape, len(adata.obs['annotation']))
        x_pixel = adata.obsm['spatial'][:,0]
        y_pixel = adata.obsm['spatial'][:,1]
        self.adj, self.adj_indices = calculate_binary_adj_matrix(coor=adata.obsm['spatial'], k_cutoff=50, model='KNN',return_indice=True)
        self.adj_hexigon_neighbor, _ = calculate_binary_adj_matrix(coor=adata.obsm['spatial'], rad_cutoff=200, model='Radius',return_indice=True)
        #adj, adj_indices = get_ground_truth_adj_matrix(coor=adata.obsm['spatial'], labels=adata.obs['annotation'], n_neighbors=6, return_indice=True)
        self.embeds = adata.obsm['X_pca']
        self.adj_indices = self.adj_indices.astype('int16')
        self.adj_hexigon_neighbor = self.adj_hexigon_neighbor.astype('float32')
        self.sample_size = adata.shape[0]
        self.label_annot = adata.obs['annotation']
        print(np.unique( self.label_annot))
        scaler = MinMaxScaler()
        self.coor = scaler.fit_transform(adata.obsm['spatial'].astype('float32'))
        print(self.embeds.shape, self.adj.shape, self.adj_indices.shape, adata.obs['annotation'].shape, self.coor.shape)

    def get_batch(self, batch_size, use_local = False):
        if use_local:
            center_idx = np.random.randint(low = 0, high = self.sample_size, size = 1)[0]
            indx = self.adj_indices[center_idx][:batch_size]
        else:
            indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        X_batch = self.embeds[indx,:]
        coor_batch = self.coor[indx,:]
        adj_batch = self.adj[indx,:][:,indx]
        adj_hexigon_batch = self.adj_hexigon_neighbor[indx,:][:,indx]
        X_neighbors_batch = [self.embeds[self.adj_indices[i]] for i in indx]
        adj_neighbors_batch = [self.adj[self.adj_indices[i],:][:,self.adj_indices[i]] for i in indx]
        return X_batch, adj_batch, adj_hexigon_batch, X_neighbors_batch, adj_neighbors_batch, coor_batch
    def load_all(self):
        return self.embeds, self.adj


def Dataset_selector(name):
    if 'acic' in name:
        return Semi_acic_sampler
    elif 'ihdp' in name:
        return Semi_ihdp_sampler
    elif 'Hirano' in name:
        return Sim_Hirano_Imbens_sampler
    elif 'quadratic' in name:
        return Sim_quadratic_sampler
    elif 'linear' in name:
        return Sim_linear_sampler
    else:
        print('Cannot find the data sampler!')
        sys.exit()

class Sim_linear_sampler(object):
    def __init__(self, N = 20000, v_dim=10, z0_dim=1, z1_dim=2, z2_dim=2, z3_dim=5, ax = 1, bx = 1):
        np.random.seed(123)
        self.sample_size = N
        self.v_dim = v_dim
        self.z0_dim = z0_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        self.ax = ax
        self.bx = bx
        self.alpha = np.ones(((z0_dim+z1_dim),1))
        self.beta_1 = np.ones((z0_dim,1)) 
        self.beta_2 = np.ones((z2_dim,1)) 
        self.data_v = np.random.normal(0, 1, size=(N, v_dim))
        self.data_x = self.get_value_x(self.data_v)
        self.data_y = self.get_value_y(self.data_v, self.data_x)
        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('float32')
        self.data_v = self.data_v.astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)
        print(np.max(self.data_x),np.max(self.data_y), np.max(self.data_v)) #8.036544 242.13437 4.350001
        print(np.min(self.data_x),np.min(self.data_y), np.min(self.data_v)) #-7.866399 -261.7619 -4.60713

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]
    
    #get x given v(or z1)
    def get_value_x(self, v):
        return np.random.normal(np.dot(v[:,:(self.z0_dim + self.z1_dim)],self.alpha), 1)

    #get y given x and v(or z1)
    def get_value_y(self, v, x):
        return np.random.normal(self.bx * x + self.ax* x**2 * (np.dot(v[:,:self.z0_dim], self.beta_1) + \
        np.dot(v[:,(self.z0_dim+self.z1_dim):(self.z0_dim+self.z1_dim+self.z2_dim)], self.beta_2)) , 1)

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Sim_quadratic_sampler(object):
    def __init__(self, N = 20000, v_dim=25, z0_dim=3, z1_dim=3, z2_dim=3, z3_dim=3, ax = 1, bx = 1):
        np.random.seed(123)
        self.sample_size = N
        self.v_dim = v_dim
        self.z0_dim = z0_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        self.ax = ax
        self.bx = bx
        self.alpha = np.ones(((z0_dim+z1_dim),1))
        self.beta_1 = np.ones((z0_dim,1)) 
        self.beta_2 = np.ones((z2_dim,1)) 
        self.data_v = np.random.normal(0, 1, size=(N, v_dim))
        self.data_x = self.get_value_x(self.data_v)
        self.data_y = self.get_value_y(self.data_v, self.data_x)
        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('float32')
        self.data_v = self.data_v.astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)
        print(np.max(self.data_x),np.max(self.data_y), np.max(self.data_v))#10.559797 156.67995 4.350001
        print(np.min(self.data_x),np.min(self.data_y), np.min(self.data_v))#-10.261807 -9.006005 -4.60713

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]
    
    #get x given v(or z1)
    def get_value_x(self, v):
        return np.random.normal(np.dot(v[:,:(self.z0_dim + self.z1_dim)],self.alpha), 0.1)

    #get y given x and v(or z1)
    def get_value_y(self, v, x):
        return np.random.normal(self.bx * x**2 + self.ax* x * (np.dot(v[:,:self.z0_dim], self.beta_1) + \
        np.dot(v[:,(self.z0_dim+self.z1_dim):(self.z0_dim+self.z1_dim+self.z2_dim)], self.beta_2)) , 0.1)

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Sim_Hirano_Imbens_sampler(object):
    def __init__(self, N = 20000, v_dim=15, z0_dim=1, z1_dim=1, z2_dim=1):
        self.data = np.loadtxt('../baselines/Imbens_sim_data.txt',usecols=range(0,17),delimiter='\t')
        self.sample_size = N
        self.v_dim = v_dim
        self.data_v = self.data[:, 0:v_dim].astype('float32')
        self.data_x = self.data[:, v_dim].reshape(-1, 1).astype('float32')
        self.data_y = (self.data[:, v_dim+1]).reshape(-1, 1).astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Semi_acic_sampler(object):
    def __init__(self, N = 20000, v_dim=117, z0_dim=1, z1_dim=1, z2_dim=1,
                path='../data/ACIC_2018', ufid='5d4cabab88b247d1b48cd38b46555b2c'):
        self.df_covariants = pd.read_csv('%s/x.csv'%path, index_col='sample_id',header=0, sep=',')
        self.df_sim = pd.read_csv('%s/scaling/factuals/%s.csv'%(path, ufid),index_col='sample_id',header=0, sep=',')
        dataset = self.df_covariants.join(self.df_sim, how='inner')
        self.data_x = dataset['z'].values.reshape(-1,1)
        self.data_y = dataset['y'].values.reshape(-1,1)
        self.data_v = dataset.values[:,:-2]
        self.data_v = self.normalize(self.data_v)

        self.sample_size = len(self.data_x)
        self.v_dim = v_dim
        self.data_v = self.data_v.astype('float32')
        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)

    def normalize(self, data):
        normal_scalar = StandardScaler()
        data = normal_scalar.fit_transform(data)
        return data

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Semi_ihdp_sampler(object):
    def __init__(self, N = 20000, v_dim=25, path='../data/IHDP_100', ufid='ihdp_npci_1'):
        self.data_all = np.loadtxt('%s/%s.csv'%(path, ufid), delimiter=',')
        self.data_x = self.data_all[:,0].reshape(-1,1)
        self.data_v = self.data_all[:,5:]
        self.data_y = self.data_all[:,1].reshape(-1,1)
        self.sample_size = len(self.data_x)
        self.v_dim = v_dim
        self.data_v = self.data_v.astype('float32')
        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]

    def load_all(self):
        return self.data_x, self.data_y, self.data_v


class Gaussian_sampler(object):
    def __init__(self, N, mean, sd=1):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.X = self.X.astype('float32')
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

    def load_all(self):
        return self.X, self.Y

#sample continuous (Gaussian) and discrete (Catagory) latent variables together
class Mixture_sampler(object):
    def __init__(self, nb_classes, N, dim, sd, scale=1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.sd = sd 
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, self.sd**2, (self.total_size,self.dim))
        #self.X_c = self.scale*np.random.uniform(-1, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes, dtype='float32')[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d)).astype('float32')
    
    def train(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim)).astype('float32')
        #X_batch_c = self.scale*np.random.uniform(-1, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes,dtype='float32')[label_batch_idx]
        return X_batch_c, X_batch_d

    def load_all(self):
        return self.X_c, self.X_d

#sample continuous (Gaussian Mixture) and discrete (Catagory) latent variables together
class Mixture_sampler_v2(object):
    def __init__(self, nb_classes, N, dim, weights=None,sd=0.5):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        np.random.seed(1024)
        if nb_classes<=dim:
            self.mean = np.random.uniform(-5,5,size =(nb_classes, dim))
            #self.mean = np.zeros((nb_classes,dim))
            #self.mean[:,:nb_classes] = np.eye(nb_classes)
        else:
            if dim==2:
                self.mean = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
            else:
                self.mean = np.zeros((nb_classes,dim))
                self.mean[:,:2] = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
        self.cov = [sd**2*np.eye(dim) for item in range(nb_classes)]
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        self.Y = np.random.choice(self.nb_classes, size=N, replace=True, p=weights)
        self.X_c = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_d = np.eye(self.nb_classes)[self.Y]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X_c[indx, :], self.X_d[indx, :], self.Y[indx, :]
        else:
            return self.X_c[indx, :], self.X_d[indx, :]

    def get_batch(self,batch_size,weights=None):
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        return self.X_c[label_batch_idx, :], self.X_d[label_batch_idx, :]
    def predict_onepoint(self,array):#return component index with max likelyhood
        from scipy.stats import multivariate_normal
        assert len(array) == self.dim
        return np.argmax([multivariate_normal.pdf(array,self.mean[idx],self.cov[idx]) for idx in range(self.nb_classes)])

    def predict_multipoints(self,arrays):
        assert arrays.shape[-1] == self.dim
        return map(self.predict_onepoint,arrays)
    def load_all(self):
        return self.X_c, self.X_d, self.label_idx
    
def softmax(x):
    """ softmax function """
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1
    x -= np.max(x, axis = 1, keepdims = True)
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

#get a batch of data from previous 50 batches, add stochastic
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data

if __name__=="__main__":
    a = Spatial_DLPFC_sampler()
    b = a.get_batch(batch_size=32)