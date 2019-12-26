import numpy as np
import pandas as pd
import os
import math
import gc
import time
import pickle
from pylab import *
from multiprocessing import Pool
import warnings 
warnings.filterwarnings('ignore')
datapath = '/media/ant2018/kdxf/fusai/'#根目录

################################### 函数function ##############################

#根据门限T，变化绝对值小于T的异常不矫正（时间、累积量参数1&2）
def threevalue(x,T):
    if x > T: 
        return x
    elif x < -T:
        return x
    else:
        return 0

#矫正部件工作时间、累积量参数，使其基本符合单调性
def preprocess_single_sample(file,coal_path):
# def preprocess_single_sample(file,dic_file,coal_path):
    train_sam = pd.read_csv(file)
    csv_name = file.split('/')[-1]
#     dic_file[csv_name] = []
    T = {'部件工作时长':50,'累积量参数1':500,'累积量参数2':100}
    for col in ['部件工作时长','累积量参数1','累积量参数2']:
        seq = list(train_sam[col].values)
        time_seq = train_sam[col]
        time_seq0 = pd.Series([seq[0]]+ seq[:-1],index = time_seq.index)

        delt = (time_seq-time_seq0).apply(lambda x : threevalue(x,T[col]))
        list_ind = delt[delt != 0].index
        val = 0
        for i in range(len(list_ind)):
            start = list_ind[i]
            try:
                end = list_ind[i+1]-1
            except:
                end = delt.shape[0]
            val = val - delt[start]
            train_sam.loc[start:end,col] = train_sam.loc[start:end,col] + val
#         dic_file[csv_name].append(val)
        try:
            time_0 = train_sam.at[list_ind[0]-1,'部件工作时长']
        except:
            time_0 = -1
#         dic_file[csv_name].append(time_0)
#         dic_file[csv_name].append(len(list_ind))
    train_sam.to_csv(coal_path,header = True,index = False)
#     return dic_file

#多线程处理上述矫正函数
def get_together0(n_cpu,filelist,func,data_type):
    
    rst = []
    pool = Pool(n_cpu)
#     dic_file = {}
    for file in filelist:
        coal_path = datapath+data_type+'_new/'+file
        file = datapath+data_type+'/'+file
        pool.apply_async(func, args = (file,coal_path)).get()
#         dic_file = pool.apply_async(func, args = (file,dic_file,coal_path)).get()
    pool.close()
    pool.join()
    
#     return dic_file

#构造基本统计特征
def create_fea(data,dic,name):
    dic[name + '_max'] = data.max()
    dic[name + '_min'] = data.min()
    dic[name + '_mean'] = data.mean()
    dic[name + '_ptp'] = data.ptp()
    dic[name + '_std'] = data.std()
    rms = math.sqrt(pow(data.mean(),2) + pow(data.std(),2))
    if name not in ['部件工作时长', '累积量参数1', '累积量参数2']:
        dic[name+'_sc'] = data.skew() #计算偏斜度
        dic[name+'_ku'] = data.kurt()#计算峰度（峭度）
        #波形因子
        dic[name+'_shapefacter'] = rms / (abs(data).mean())
        #峰值因子
        dic[name+'_peekfacter'] =(max(data)) / rms
    return dic

#对单个样本文件构造若干样本
def process_sample_single(file,per):
    data = pd.read_csv(file)
    lifemax = data['部件工作时长'].max()
    data = data[data['部件工作时长'] <= lifemax*per]
    dic_sam = {'设备id': os.path.basename(file)+'_'+str(per),
               '开关1_sum':data['开关1信号'].sum(),
               '开关2_sum':data['开关2信号'].sum(),
               '告警1_sum':data['告警信号1'].sum(),
               '设备类型':data['设备类型'][0],
               'life':lifemax-data['部件工作时长'].max()
              }
    for i in ['部件工作时长', '累积量参数1', '累积量参数2',
              '转速信号1','转速信号2','压力信号1','压力信号2',
              '温度信号','流量信号','电流信号']:
        dic_sam = create_fea(data[i],dic_sam,i)
    sam = pd.DataFrame(dic_sam, index=[0])  
    return sam

#多线程上述样本构造函数
def get_together(n_cpu,filelist,per_list,func,data_type,keep_per):
    
    rst = []
    pool = Pool(n_cpu)
    for file in filelist:
        file = datapath+data_type+'/'+file
        for per in per_list:
            rst.append(pool.apply_async(func, args = (file,per)))
    pool.close()
    pool.join()
    rst = [i.get() for i in rst]
    tv_features = pd.concat(rst,axis = 0,ignore_index = True)
    if keep_per:
        tv_features['percent'] = tv_features['设备id'].apply(lambda x : x.split('_')[1])
    tv_features['设备id'] = tv_features['设备id'].apply(lambda x : x.split('_')[0])
    cols = tv_features.columns.tolist()
    for col in ['设备id','life']:
        cols.remove(col)
    cols=['设备id']+cols+['life']
    tv_features = tv_features.reindex(columns = cols)
    
    return tv_features

################################## 预处理矫正序列异常偏移 ##############################

train_files = os.listdir(datapath+'train')
test_files = os.listdir(datapath+'test2')

#预处理时间序列的异常偏移
##偏移校正后会生成新的文件存储在根目录下的train_new和test2_new文件夹下
get_together0(12,train_files,preprocess_single_sample,'train')
print('train时间偏移处理完毕！')
get_together0(12,test_files,preprocess_single_sample,'test2')
print('test时间偏移处理完毕！')

################################# 构造样本 ###################################

perlist_train = [0.50,0.54,0.58,0.65,0.72,0.77]
# perlist_train = [0.45,0.55,0.63,0.75,0.85]
perlist_test = [1]
train = get_together(24,train_files,perlist_train,process_sample_single,'train_new',keep_per = True)
test = get_together(24,test_files,perlist_test,process_sample_single,'test2_new',keep_per = True)
train.to_csv(datapath+'train.csv',header = True,index = False)
test.to_csv(datapath+'test.csv',header = True,index = False)
#构造的训练集测试集存储在根目录下