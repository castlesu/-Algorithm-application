import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
fontprop = fm.FontProperties(fname="Jalnan.ttf",size =15)


def File_Open():
    data = pd.read_csv('seoul.txt', sep='\t', encoding='utf-8', index_col='성별')
    del data['행정구역별']
#    print(data)

    return data

def File_Sum():
    data =File_Open()
#    print(data)
    sex = data.loc[['여자','남자','계'],:]
    sex.loc['계',:] = [data.loc[['계'],:].sum()]
    sex.loc['남자', :] = [data.loc[['남자'], :].sum()]
    sex.loc['여자', :] = [data.loc[['여자'], :].sum()]
    sex.head()
    sum_data = sex.drop_duplicates()

#    print(sum_data)
    return sum_data

def File_Graph():
    data =File_Sum()
    data_array = np.array(data.values)

    f_data =data_array[0]
    m_data =data_array[1]
    t_data =data_array[2]

    fig =plt.figure(3,[20,5])
    female = fig.add_subplot(1,3,1)
    male = fig.add_subplot(1,3,2)
    total = fig.add_subplot(1,3,3)

    female.bar(range(len(f_data)), f_data)
    female.set_xlabel('여자',fontproperties = fontprop)
    male.bar(range(len(m_data)),m_data)
    male.set_xlabel('남자',fontproperties = fontprop)
    total.bar(range(len(t_data)),t_data)
    total.set_xlabel('계',fontproperties = fontprop)

    plt.show()

    return plt.show()

def File_Calculate():
    data = File_Sum()
    data_array = np.array(data.values)
    name = ['여자','남자','계']

    for i in range(len(data_array)) :
        join_data=" ".join(map(str,data_array[i]))
        print('%s :' %name[i], join_data)
        print('%s 총합 : ' %name[i], np.sum(data_array[i]))
        print('%s 평균 : ' % name[i], int(np.mean(data_array[i])))
        print('%s 분산 : ' % name[i], int(np.ceil(np.var(data_array[i]))))
        print('\n')

    return

File_Graph() #그래프 창 닫으면 cal함수 실행
File_Calculate()