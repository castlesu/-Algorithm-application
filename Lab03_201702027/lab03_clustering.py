import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


import random
from collections import OrderedDict
import matplotlib.font_manager as fm
fontprop = fm.FontProperties(fname="Jalnan.ttf",size =15)

def File_Open():
    data = pd.read_csv('covid-19.txt',sep='\t', encoding='utf-8')
    data = data.loc[:,['사망률', '완치율']]
    # data = np.array(data)
    scaler = MinMaxScaler()
    data[:] = scaler.fit_transform(data[:])

    return data

def Draw_Graph(data, labels):
    plt.figure()
    plt.scatter(data[:,0],data[:,1],c=labels,cmap='rainbow')
    plt.show()
    return

def DBSCAN(data):
    data= np.array(data)
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(data)
    labels =clustering.labels_

    return data, labels

def AgglomerativeClustering(data):
    data = np.array(data)
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=8,linkage='complete').fit(data)
    labels =clustering.labels_

    return data, labels

class KMeans:
    def __init__(self,data,n):
        self.data = np.array(data)
        self.n = n
        self.cluster = OrderedDict()

    def init_center(self):
        index = random.randint(0,self.n)
        index_list = []
        for i in range(self.n):
            while index in index_list:
                index = random.randint(0,self.n)
            index_list.append(index)
            self.cluster[i] = {'center': self.data[index], 'data': []}

    def clustering(self,cluster):
        center = []
        com_data=[[] for i in range(len(cluster.keys()))]
        for i in range(len(self.data)):
            eucl_data = []
            for j in range(len(cluster.keys())):
                center.append(self.cluster[j].get('center'))
                euclidean = np.linalg.norm(center[j]-self.data[i])
                eucl_data.append(euclidean)
            for index in range(len(cluster.keys())):
                if(np.argmin(eucl_data) == index):
                    com_data[index].append(self.data[i])
        for i in range(len(cluster.keys())):
            self.cluster[i]['data'] = com_data[i]

        return self.cluster

    def update_center(self):
        cen_data = [[] for i in range(self.n)]
        cen_data1 = []
        cen_data2 = []
        compare = [[] for i in range(self.n)]

        for i in range(self.n):# 기존 센터값
            cen_data1.append(self.cluster[i].get('center'))
        print('cen_data1',cen_data1)

        for i in range(self.n):
            data_avg=np.average(self.cluster[i]['data'], axis=0)
            cen_data[i].append(data_avg)
        for i in range(self.n):
            self.cluster[i]['center'] = cen_data[i]

        for i in range(self.n):  # 바뀐 센터값
            cen_data2.append(self.cluster[i].get('center'))
        cen_data2 = [elem for twd in cen_data2 for elem in twd]
        print('cen_data2',cen_data2)

        for i in range(self.n): #센터값 비교
            compare[i].append(cen_data1[i] == cen_data2[i])
        print('compare_array',compare)
        compare = np.array(compare).flatten()
        print(compare)

        return self.cluster , compare

    def update(self):
        while True:
            new_cluster, compare = self.update_center()
            chk = compare.all()
            print(chk)
            if(chk==True):
                print('while true',chk)
                break
            self.clustering(new_cluster)

        return

    def fit(self):
        self.init_center()
        self.cluster = self.clustering(self.cluster)
        self.update()

        result, labels = self.get_result(self.cluster)
        Draw_Graph(result,labels)

    def get_result(self,cluster):
        result = []
        labels = []
        for key, value in cluster.items():
            for item in value['data']:
                labels.append(key)
                result.append(item)

        return np.array(result), labels

if __name__ == '__main__':
    KMeans(File_Open(), 8).fit()
    data,labels = DBSCAN(File_Open())
    Draw_Graph(data,labels)

    data ,labels =AgglomerativeClustering(File_Open())
    Draw_Graph(data,labels)
