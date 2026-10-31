import os
from scipy import stats
import numpy as np


def tTest(data1, data2):
# print(data1.shape, data2.shape)
    re_data2 = np.zeros((data1.shape))
##    print(re_data2)
    i = 0
    while i < data1.shape[0]:
        re_data2[i:i+3] = data2[int(i/3)]
        i += 3
##    print(re_data2.shape, re_data2)        
    data11 = np.divide(data1, 100)
    data12 = np.divide(re_data2, 100)
    if ((data11 == data12).all()):
        return str(' ')
    else:
        #print('data1',data1)
        #print('data2',data2)
        t,p = stats.tTest(data11, data12)
        mean_data1 = np.mean(data11)
        mean_data2 = np.mean(data12)
        #print(t,p,mean_data2,mean_data1)
        if p < 0.05:
            if mean_data1 > mean_data2:
                return str('+')
            else:
                return str('-')
        elif p >= 0.05:
            return str(' ')
        else:
            return str('/')


def wilcoxonT(data1,data2):
##    print(data1.shape, data2.shape)
    re_data2=np.zeros((data1.shape))
##    print(re_data2)
    i=0
    while i <  data1.shape[0]:
        re_data2[i:i+3]=data2[int(i/3)]
        i+=3
##    print(re_data2.shape, re_data2)        
    data11=np.divide(data1,100)
    data12=np.divide(re_data2,100)
    if ((data11==data12).all()):
        return str(' ')
    else:
        #print('data1',data1)
        #print('data2',data2)
        t,p = stats.wilcoxon(data11, data12)
        mean_data1=np.mean(data11)
        mean_data2=np.mean(data12)
        #print(t,p,mean_data2,mean_data1)
        if p<0.05:
            if mean_data1 > mean_data2:
                return str('+')
            else:return str('--')
        elif p>=0.05:
            return str('=')
        else:
            return str('/')


def wilcoxonT_unpaired(data1, data2):
##    print(data1.shape, data2.shape)
    # re_data2 = np.zeros((data1.shape))
##    print(re_data2)
    # i = 0
    # while i < data1.shape[0]:
    #     re_data2[i:i+3] = data2[int(i/3)]
    #     i += 3
##    print(re_data2.shape, re_data2)        
    data11 = np.divide(data1, 100)
    data12 = np.divide(data2, 100)
    if ((data11==data12).all()):
        return str(' ')
    else:
        #print('data1',data1)
        #print('data2',data2)
        t, p = stats.ranksums(data11, data12)
        mean_data1 = np.mean(data11)
        mean_data2 = np.mean(data12)
        #print(t,p,mean_data2,mean_data1)
        if p < 0.05:
            if mean_data1 > mean_data2:
                return str('+')
            else:
                return str('--')
        elif p >= 0.05:
            return str('=')
        else:
            return str('/')
