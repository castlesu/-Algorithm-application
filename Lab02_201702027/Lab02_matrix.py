import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.preprocessing import MinMaxScaler
fontprop = fm.FontProperties(fname="Jalnan.ttf",size =15)



def File_Open():
    data = pd.read_csv('seoul_tax.txt', sep='\t', encoding='utf-8')
    del data['자치구별']
    data = np.array(data.values)
    name = ' '

    return data , name

def Cosine_Distance(data):
    data , names= data
    CosineDistance = 1 - cs(data)

    print(CosineDistance)
    name = '코사인 '+names

    return CosineDistance, name


def Manhattan_Distance(data):
    data , names= data
    ManhattanDistance = np.zeros((data.shape[0],data.shape[0]),dtype=float)
    for i in range(data.shape[0]):
        for j in range(i,data.shape[0]):
            sub = data[i] - data[j]
            manhattan = np.sum(np.abs(sub))
            ManhattanDistance[i,j] = manhattan
            ManhattanDistance[j,i] = manhattan

    # print(ManhattanDistance)
    name = '맨하탄 ' + names
    return ManhattanDistance , name

def Euclidean_Distance(data_def):
    data , names = data_def
    print(data)
    b = data.reshape(data.shape[0], 1, data.shape[1])

    EuclideanDistance = np.sqrt(np.einsum('ijk, ijk->ij', data - b, data - b))

    # print(EuclideanDistance)
    print(EuclideanDistance)
    name = '유클리디안 ' + names
    print(names)
    return EuclideanDistance, name

def Data_Norm():
    data,name = File_Open()
    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)
    norm_data = data
    # print(norm_data)

    norm_name = ' 정규화'

    return norm_data,norm_name


def Graph(data_def):
    data , name = data_def
    plt.pcolor(data)
    plt.xlabel('%s'%name,fontproperties = fontprop)
    plt.colorbar()

    plt.show()

    return


def main():
   # Graph(Cosine_Distance(File_Open()))
   # Graph(Euclidean_Distance(File_Open()))
   # Graph(Manhattan_Distance(File_Open()))
   #
   # Graph(Euclidean_Distance(Data_Norm()))
   # Graph(Cosine_Distance(Data_Norm()))
   # Graph(Manhattan_Distance(Data_Norm()))
   Euclidean_Distance(File_Open())

if __name__ == '__main__':
    main()