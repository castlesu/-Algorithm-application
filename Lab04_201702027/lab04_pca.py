import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm
fontprop = fm.FontProperties(fname="Jalnan.ttf",size =15)

def File_Open():
    data = pd.read_csv('seoul_student.txt',sep='\t', encoding='utf-8')
    data = np.array(data)
    scaler = MinMaxScaler()
    data[:] = scaler.fit_transform(data[:])

    return data

def Draw_Graph(data):
    plt.figure()
    plt.scatter(data,[0]*len(data),cmap='rainbow')
    plt.show()
    return

def Draw_Ori_Graph(data):
    plt.figure()
    plt.scatter(data[:,0],data[:,1],cmap='rainbow')
    plt.show()
    return

def Get_Covariance_Matrix(data):
    data =data
    data_min_avg2 = [[] for i in range(data.shape[0])]
    avg = []
    avg1 =0
    avg2 =0

    for i in range(data.shape[0]):
        avg1 += data[i][0]
        avg2 += data[i][1]

    avg1 = avg1/(data.shape[0])
    avg2 = avg2/(data.shape[0])
    avg.append(avg1)
    avg.append(avg2)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_min_avg2[i].append(data[i][j]- avg[j])

    data_min_avg2 = np.array(data_min_avg2)
    cov= (np.dot((data_min_avg2.T),(data_min_avg2)))/(len(data)-1)

    return cov

def Get_Eigen(cov):
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov)

    return eig_val_cov,eig_vec_cov

def E_Sort(val,vec,dim):
    eig_val = val
    eig_vec = vec
    dim = dim
    idx = np.argsort(-eig_val)
    # sort_val = eig_val[idx]
    sort_vec=eig_vec[:,idx]
    sel_vec = []
    # print(sel_vec)
    for i in range(dim):
        sel_vec.append(sort_vec[i])

    return sel_vec

def Pca(data, dim):
    data=data
    dim = dim
    cov =Get_Covariance_Matrix(data)
    eig_val , eig_vec =Get_Eigen(cov)
    pca_vec = E_Sort(eig_val,eig_vec,dim)
    pca_data = np.array(pca_vec).dot(data.T)
    pca_data = pca_data.T

    Draw_Graph(pca_data)

    return

def Sklearn_Pca(data,dim):
    data = data
    dim = dim
    pca = PCA(n_components=dim)
    pca_Data =pca.fit_transform(data)

    Draw_Graph(pca_Data)

    return


if __name__ == '__main__':
    reduce_dim = 1
    data = File_Open()
    Draw_Ori_Graph(data)

    Pca(copy.deepcopy(data), reduce_dim)
    Sklearn_Pca(copy.deepcopy(data),reduce_dim)
