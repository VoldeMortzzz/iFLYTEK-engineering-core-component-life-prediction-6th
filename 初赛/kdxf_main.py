import numpy as np
import pandas as pd
import os
import math
import gc
import time
import pickle
from multiprocessing import Pool
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import warnings 
import lightgbm as lgb
warnings.filterwarnings('ignore')
datapath = '/media/ant2018/kdxf/new/'#根目录

################################### 函数function #################################
#根据门限T，变化绝对值小于T的异常不矫正（时间、累积量参数1&2）
def threevalue(x,T):
    if x > T: 
        return x
    elif x < -T:
        return x
    else:
        return 0

#得到时间、累加量参数12预处理前异常波动的特征（第一个波动时间、波动数量、偏移累加值）    
def create_modify_fea_single(file,process_file,dic_file,perlist):
    
    train_sam = pd.read_csv(file)
    sam_new = pd.read_csv(process_file)
    life_max = sam_new['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        per_indexes[per] = np.max(sam_new[sam_new['部件工作时长'] <= life_max*per].index.to_list())

    T = {'部件工作时长':50,'累积量参数1':500,'累积量参数2':100}
    for col in ['部件工作时长','累积量参数1','累积量参数2']:
        seq = list(train_sam[col].values)
        time_seq = train_sam[col]
        time_seq0 = pd.Series([seq[0]]+ seq[:-1],index = time_seq.index)

        delt = (time_seq-time_seq0).apply(lambda x : threevalue(x,T[col]))
        list_ind = delt[delt != 0].index
        list_ind = np.array(list_ind)
        val = 0
        for per in perlist:#delt/t0/num
            try:
                ind = list_ind[list_ind <= per_indexes[per]].min()
                t0 = sam_new.at[ind,'部件工作时长']
            except:
                t0 = -1
            dic_file[csv_name+'_'+str(per)].append(delt[:per_indexes[per]].sum())
            dic_file[csv_name+'_'+str(per)].append(t0)
            dic_file[csv_name+'_'+str(per)].append(len(list_ind[list_ind <= per_indexes[per]]))
            
    return dic_file

#时间、累加量变化曲线分段斜率特征
def create_k_fea_single(file,dic_file,perlist,col):
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/6) for i in range(1,7)]
        inds = [0] + inds
        for i in range(6):
            start = inds[i]
            end = inds[i+1]
            value = sam.at[end,col] - sam.at[start,col]
            delt_ind = end-start
            dic_file[csv_name+'_'+str(per)].append(value/delt_ind)            

    return dic_file

#温度分段统计特征
def create_errornum_wendu_single(file,dic_file,perlist,col):
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/4) for i in range(1,5)]
        inds = [0] + inds
        num1 = 0
        num2 = 0
        for i in range(4):
            start = inds[i]
            end = inds[i+1]
            mean = sam.loc[start:end,col].mean()
            dic_file[csv_name+'_'+str(per)].append(mean)
            std = sam.loc[start:end,col].std()
            dic_file[csv_name+'_'+str(per)].append(std)
            try:
                num1 = num1 + (sam.loc[start:end,col] >= mean + std*1.8).value_counts()[True]
            except:
                num1 = num1
            try:
                num2 = num2 + (sam.loc[start:end,col] <= mean - std*1.8).value_counts()[True]
            except:
                num2 = num2
        dic_file[csv_name+'_'+str(per)].append(num1)
        dic_file[csv_name+'_'+str(per)].append(num2)

    return dic_file

#压力分段统计特征
def create_errornum_yali_single(file,dic_file,perlist,col):
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    mode = sam[col].mode()
    if col == '压力信号1':
        mode = mode[mode < 100].mean()
    else:
        mode = mode.mean()
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/4) for i in range(1,5)]
        inds = [0] + inds
        num2 = 0
        for i in range(4):
            start = inds[i]
            end = inds[i+1]
            mean = sam.loc[start:end,col].mean()
            dic_file[csv_name+'_'+str(per)].append(mean)
            std = sam.loc[start:end,col].std()
            dic_file[csv_name+'_'+str(per)].append(std)
            if col == '压力信号1':
                hold = mode-10
            else:
                hold = 150
            try:
                num2 = num2 + (sam.loc[start:end,col] <= hold).value_counts()[True]
            except:
                num2 = num2
        dic_file[csv_name+'_'+str(per)].append(num2)
        dic_file[csv_name+'_'+str(per)].append(mode)

    return dic_file

#流量分段统计特征
def create_errornum_liuliang_single(file,dic_file,perlist,col):
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/4) for i in range(1,5)]
        inds = [0] + inds
        num2 = 0
        for i in range(4):
            start = inds[i]
            end = inds[i+1]
            mean = sam.loc[start:end,col].mean()
            dic_file[csv_name+'_'+str(per)].append(mean)
            std = sam.loc[start:end,col].std()
            dic_file[csv_name+'_'+str(per)].append(std)

    return dic_file

#转速分段统计特征
def create_errornum_zhuansu_single(file,dic_file,perlist,col):
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    mode = sam[col].mode()
    mode = mode[mode < 6000].mean()
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/4) for i in range(1,5)]
        inds = [0] + inds
        num2 = 0
        for i in range(4):
            start = inds[i]
            end = inds[i+1]
            mean = sam.loc[start:end,col].mean()
            dic_file[csv_name+'_'+str(per)].append(mean)
            std = sam.loc[start:end,col].std()
            dic_file[csv_name+'_'+str(per)].append(std)
            hold = mode-1000
            try:
                num2 = num2 + (sam.loc[start:end,col] <= hold).value_counts()[True]
            except:
                num2 = num2
        dic_file[csv_name+'_'+str(per)].append(num2)
        dic_file[csv_name+'_'+str(per)].append(mode)

    return dic_file

#二值类特征（开关、告警）分段统计特征
def create_2zhi_single(file,dic_file,perlist,col):
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/6) for i in range(1,7)]
        inds = [0] + inds
        num2 = 0
        for i in range(6):
            start = inds[i]
            end = inds[i+1]
            he =  sam.loc[start:end,col].sum()
            dic_file[csv_name+'_'+str(per)].append(he)
            mean = sam.loc[start:end,col].mean()
            dic_file[csv_name+'_'+str(per)].append(mean)
            std = sam.loc[start:end,col].std()
            dic_file[csv_name+'_'+str(per)].append(std)

    return dic_file

#together1、2函数用于多线程处理上述特征
def get_together1(n_cpu,filelist,func,data_type,perlist):
    
    rst = []
    pool = Pool(n_cpu)
    dic_file = {}
    for f in filelist:
        file = datapath+data_type+'/'+f
        process_file = datapath+data_type+'_new/'+f
        dic_file = pool.apply_async(func, args = (file,process_file,dic_file,perlist)).get()
    pool.close()
    pool.join()
    
    return dic_file

def get_together2(n_cpu,filelist,func,data_type,perlist,col):
    
    rst = []
    pool = Pool(n_cpu)
    dic_file = {}
    for f in filelist:
        file = datapath+data_type+'_new/'+f
        dic_file = pool.apply_async(func, args = (file,dic_file,perlist,col)).get()
    pool.close()
    pool.join()
    
    return dic_file

#各merge函数用于合并train、test与构造出的各类特征（字典形式存储）
def dic_merge_origin(data,dic):
    dic = pd.DataFrame(pd.Series(dic),columns = ['values'])
    cols = ['部件工作时长_delt','部件工作时长_t0','部件工作时长_enum',
        '累积量参数1_delt','累积量参数1_t0','累积量参数1_enum',
        '累积量参数2_delt','累积量参数2_t0','累积量参数2_enum']
    for i,col in enumerate(cols):
        dic[col] = dic['values'].apply(lambda x:x[i])
    del dic['values']
    dic['设备id'] = dic.index
    dic = dic.reset_index(drop = True)
    dic['percent'] = dic['设备id'].apply(lambda x : x.split('_')[1])
    dic['设备id'] = dic['设备id'].apply(lambda x : x.split('_')[0])
    data = pd.merge(data,dic,on = ['设备id','percent'],how = 'left')
    
    return data

def dic_merge_k(dic,data,col):
    values = pd.DataFrame(dic.values(),columns = [col+'_k'+str(i) for i in range(1,7)])
    values_c = values.copy()
    values[col+'k_mean'] = values_c.mean(axis = 1)
#     values[col+'k_std'] = values_c.std(axis = 1)
#     values[col+'k_max'] = values_c.max(axis = 1)
#     values[col+'k_min'] = values_c.min(axis = 1)
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    
    return data

def dic_merge_errornum1(dic,data,col):
    cols = []
    for i in range(1,5):
        cols.append(col+'_mean'+str(i))
        cols.append(col+'_std'+str(i))
    values = pd.DataFrame(dic.values(),columns = cols+[col+'n_err+',col+'n_err-'])
    values_c = values.copy()
    values[col+'mean_std'] = values_c[[col+'_mean'+str(i) for i in range(1,5)]].std(axis = 1)
    values[col+'_errsum'] = values_c[[col+'n_err+',col+'n_err-']].sum(axis = 1)
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    del data[col+'n_err-']
    
    return data

def dic_merge_errornum2(dic,data,col):
    cols = []
    for i in range(1,5):
        cols.append(col+'_mean'+str(i))
        cols.append(col+'_std'+str(i))
    values = pd.DataFrame(dic.values(),columns = cols+[col+'_n_err-',col+'_mode'])
    values_c = values.copy()
    values[col+'mean_std'] = values_c[[col+'_mean'+str(i) for i in range(1,5)]].std(axis = 1)
    values[col+'std_mean'] = values_c[[col+'_std'+str(i) for i in range(1,5)]].mean(axis = 1)
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    
    return data

def dic_merge_meanstd(dic,data,col):
    cols = []
    for i in range(1,5):
        cols.append(col+'_mean'+str(i))
        cols.append(col+'_std'+str(i))
    values = pd.DataFrame(dic.values(),columns = cols)
    values_c = values.copy()
    values[col+'mean_std'] = values_c[[col+'_mean'+str(i) for i in range(1,5)]].std(axis = 1)
    values[col+'std_mean'] = values_c[[col+'_std'+str(i) for i in range(1,5)]].mean(axis = 1)
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
#     values = values.drop([col+'_std'+str(i) for i in range(1,5)],axis = 1)
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    
    return data

def dic_merge_sumratestd(dic,data,col):
    cols = []
    for i in range(1,7):
        cols.append(col+'_sum'+str(i))
        cols.append(col+'_rate'+str(i))
        cols.append(col+'_std'+str(i))
    values = pd.DataFrame(dic.values(),columns = cols)
    values_c = values.copy()
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
#     values = values.drop([col+'_std'+str(i) for i in range(1,5)],axis = 1)
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    
    return data

#用于构造分段统计特征的相邻段变化量特征
def deltvalues(data):
    out = []
    for i,row in data.iterrows():
        seq1 = row.values[1:]
        seq2 = row.values[:-1]
        seq_delt = seq1-seq2
        out.append(seq_delt)
    return out

#lgb训练函数
def lgb_train_pre(train_x,train_y,test,metric,log,drop_feas):
    
    train_x = train_x.drop(drop_feas,axis = 1)
    test = test.drop(drop_feas,axis = 1)
    kf = model_selection.KFold(n_splits = 5,random_state = 2019,shuffle = True)
    paras_lgb = {'boosting_type':'gbdt',
                 'objective':'regression_'+metric,
                 'random_state':2019,
                 'num_leaves':41,
                 'learning_rate':0.01,
                 'n_estimators':5000,#40000
                 'max_depth': -1, #6
                 'feature_fraction':0.8,
                 'bagging_fraction':0.7,
                 'bagging_freq':2,
#                  'min_child_samples': 30,
#                  'min_child_weight': 0.001,
                 'reg_alpha':0, 
                 'reg_lambda':2}

    lgbr = lgb.LGBMRegressor(**paras_lgb)
    categorical_feature = ['设备类型']#,'time_isminus']
    pre_train = pd.Series(index = train_y.index)
    pre_test = []
    eval_m = {'l1':'mae','l2':'rmse'}
    if log:
        train_y = train_y.apply(lambda x: math.log(x+1,2))
    for i,(train_index,test_index) in enumerate(kf.split(train)):
        x_train = train_x.loc[train_index,:]
        y_train = train_y[train_index]
        x_test = train_x.loc[test_index,:]
        y_test = train_y[test_index]
        lgbr.fit(x_train,y_train,eval_set = [(x_test,y_test)],eval_metric = eval_m[metric],early_stopping_rounds = 300,
                 verbose = 300,categorical_feature = categorical_feature)
        pre_train[test_index] = lgbr.predict(x_test)
        pre_test.append(lgbr.predict(test))

    pre_test = np.array(pre_test)
    if log:
        pre_test = np.power(2,pre_test)-1
        pre_train = pre_train.apply(lambda x:np.power(2,x)-1)
    pre_test = np.mean(pre_test,axis = 0)
    
    feas = train_x.columns
    imps = lgbr.feature_importances_
    fea_imp = pd.DataFrame(pd.Series(feas),columns = ['feas'])
    fea_imp['imp'] = imps
    fea_imp = fea_imp.sort_values(by = 'imp',ascending = False)
    return pre_train,pre_test,fea_imp

########################## 特征工程1:从文件中构造各样本各特征，存储在dic:｛key-[]｝ ######################

#文件列表
train_files = os.listdir(datapath+'train_new')
test_files = os.listdir(datapath+'test1_new')
#文件分隔比例
perlist_train = [0.45,0.55,0.63,0.75,0.85]
perlist_test = [1]

#构造to、enum、delt特征
dic_file_train = get_together1(12,train_files,create_modify_fea_single,'train',perlist_train)
dic_file_test = get_together1(12,test_files,create_modify_fea_single,'test1',perlist_test)

#count
dic_k1 = {}
for file in iter(train_files):
    sam = pd.read_csv(datapath+'train_new/'+file)
    for per in iter(perlist_train):
        time_len = sam['部件工作时长'].max()*per
        dic_k1[file+'_'+str(per)] = sam[sam['部件工作时长'] <= time_len].shape[0]
train_k = pd.DataFrame(pd.Series(dic_k1),columns = ['count'])
train_k['设备id'] = train_k.index
train_k['percent'] = train_k['设备id'].apply(lambda x : x.split('_')[1])
train_k['设备id'] = train_k['设备id'].apply(lambda x : x.split('_')[0])
train_k.reset_index(drop = True,inplace = True)

dic_k2 = {}
for file in iter(test_files):
    sam = pd.read_csv(datapath+'test1_new/'+file)
    for per in iter(perlist_test):
        time_len = sam['部件工作时长'].max()*per
        dic_k2[file+'_'+str(per)] = sam[sam['部件工作时长'] <= time_len].shape[0]
test_k = pd.DataFrame(pd.Series(dic_k2),columns = ['count'])
test_k['设备id'] = test_k.index
test_k['percent'] = test_k['设备id'].apply(lambda x : x.split('_')[1])
test_k['设备id'] = test_k['设备id'].apply(lambda x : x.split('_')[0])
test_k.reset_index(drop = True,inplace = True)

#构造时间斜率特征
dic_time_train = get_together2(12,train_files,create_k_fea_single,'train',perlist_train,'部件工作时长')
dic_time_test = get_together2(12,test_files,create_k_fea_single,'test1',perlist_test,'部件工作时长')

#温度
dic_wendu_train = get_together2(12,train_files,create_errornum_wendu_single,'train',perlist_train,'温度信号')
dic_wendu_test = get_together2(12,test_files,create_errornum_wendu_single,'test1',perlist_test,'温度信号')

#压力信号1
dic_yali1_train = get_together2(12,train_files,create_errornum_yali_single,'train',perlist_train,'压力信号1')
dic_yali1_test = get_together2(12,test_files,create_errornum_yali_single,'test1',perlist_test,'压力信号1')

#压力信号2
dic_yali2_train = get_together2(12,train_files,create_errornum_yali_single,'train',perlist_train,'压力信号2')
dic_yali2_test = get_together2(12,test_files,create_errornum_yali_single,'test1',perlist_test,'压力信号2')

#流量信号
dic_liul_train = get_together2(12,train_files,create_errornum_liuliang_single,'train',perlist_train,'流量信号')
dic_liul_test = get_together2(12,test_files,create_errornum_liuliang_single,'test1',perlist_test,'流量信号')

#转速信号1
dic_zhs1_train = get_together2(12,train_files,create_errornum_zhuansu_single,'train',perlist_train,'转速信号1')
dic_zhs1_test = get_together2(12,test_files,create_errornum_zhuansu_single,'test1',perlist_test,'转速信号1')

#电流信号
dic_I_train = get_together2(12,train_files,create_errornum_liuliang_single,'train',perlist_train,'电流信号')
dic_I_test = get_together2(12,test_files,create_errornum_liuliang_single,'test1',perlist_test,'电流信号')

#告警信号
dic_gaojing_train = get_together2(12,train_files,create_2zhi_single,'train',perlist_train,'告警信号1')
dic_gaojing_test = get_together2(12,test_files,create_2zhi_single,'test1',perlist_test,'告警信号1')

############################## 特征工程2:拼接各特征dict到train/test ###############################

train = pd.read_csv(datapath+'train.csv',header = 0)
test = pd.read_csv(datapath+'test.csv',header = 0)
train['percent'] = train['percent'].apply(str)
test['percent'] = test['percent'].apply(str)

# #is_minus 特征
# with open(datapath+'fuzhiID_train.pkl','rb') as f:
#     id_train = pickle.load(f) 
# with open(datapath+'fuzhiID_test.pkl','rb') as f:
#     id_test = pickle.load(f) 
# train['time_isminus'] = train['设备id'].apply(lambda x: 1 if x in id_train else 0)
# test['time_isminus'] = test['设备id'].apply(lambda x: 1 if x in id_test else 0)

#去掉异常样本
drop_sam = ['558df5deb7c15a6b8665.csv',
            '69d0c0d1b3a2308fa7fe.csv',
            '27b553e8ea6b0c30c2ad.csv']
train = train[(1-train['设备id'].isin(drop_sam)).astype(bool)].reset_index(drop = True)

#构造to、enum、delt特征
train = dic_merge_origin(train,dic_file_train)
test = dic_merge_origin(test,dic_file_test)

#count
train = pd.merge(train,train_k,on = ['设备id','percent'],how = 'left')
test = pd.merge(test,test_k,on = ['设备id','percent'],how = 'left')

#构造时间斜率特征
train = dic_merge_k(dic_time_train,train,'time')
test = dic_merge_k(dic_time_test,test,'time')

#温度
train = dic_merge_errornum1(dic_wendu_train,train,'wendu')
train = train.drop(['wendu'+'_std'+str(i) for i in range(1,5)],axis = 1)
test = dic_merge_errornum1(dic_wendu_test,test,'wendu')
test = test.drop(['wendu'+'_std'+str(i) for i in range(1,5)],axis = 1)

#压力信号1
train = dic_merge_errornum2(dic_yali1_train,train,'yali1')
train = train.drop(['yali1mean_std','yali1std_mean','yali1_n_err-'],axis = 1)
test = dic_merge_errornum2(dic_yali1_test,test,'yali1')
test = test.drop(['yali1mean_std','yali1std_mean','yali1_n_err-'],axis = 1)

#压力信号2
train = dic_merge_errornum2(dic_yali2_train,train,'yali2')#'yali2_mode','yali2_n_err-'
train = train.drop(['yali2mean_std','yali2std_mean','yali2_n_err-']+['yali2'+'_mean'+str(i) for i in range(1,5)]+['yali2'+'_std'+str(i) for i in range(1,5)],axis = 1)
test = dic_merge_errornum2(dic_yali2_test,test,'yali2')
test = test.drop(['yali2mean_std','yali2std_mean','yali2_n_err-']+['yali2'+'_mean'+str(i) for i in range(1,5)]+['yali2'+'_std'+str(i) for i in range(1,5)],axis = 1)

#流量信号
train = dic_merge_meanstd(dic_liul_train,train,'liul')
test = dic_merge_meanstd(dic_liul_test,test,'liul')

#转速信号1
train = dic_merge_errornum2(dic_zhs1_train,train,'zhs1')
train = train.drop(['zhs1std_mean','zhs1mean_std','zhs1_mode','zhs1_n_err-']+['zhs1'+'_std'+str(i) for i in range(1,5)],axis = 1)
test = dic_merge_errornum2(dic_zhs1_test,test,'zhs1')
test = test.drop(['zhs1std_mean','zhs1mean_std','zhs1_mode','zhs1_n_err-']+['zhs1'+'_std'+str(i) for i in range(1,5)],axis = 1)

#电流信号
train = dic_merge_meanstd(dic_I_train,train,'I')
train = train.drop(['Istd_mean','Imean_std']+['I'+'_mean'+str(i) for i in range(1,5)],axis = 1)
test = dic_merge_meanstd(dic_I_test,test,'I')
test = test.drop(['Istd_mean','Imean_std']+['I'+'_mean'+str(i) for i in range(1,5)],axis = 1)

#告警信号
train = dic_merge_sumratestd(dic_gaojing_train,train,'gj')#['gj'+'_rate'+str(i) for i in range(1,7)]+
train = train.drop(['gj'+'_std'+str(i) for i in range(1,7)],axis = 1)
test = dic_merge_sumratestd(dic_gaojing_test,test,'gj')
test = test.drop(['gj'+'_std'+str(i) for i in range(1,7)],axis = 1)

#告警信号的分段delt特征
data_delt_train = deltvalues(train[['gj'+'_sum'+str(i) for i in range(1,7)]])
data_delt_train = pd.DataFrame(data_delt_train,columns = ['gj_deltsum'+str(i) for i in range(1,6)])
data_delt_test = deltvalues(test[['gj'+'_sum'+str(i) for i in range(1,7)]])
data_delt_test = pd.DataFrame(data_delt_test,columns = ['gj_deltsum'+str(i) for i in range(1,6)])
train = pd.concat([train,data_delt_train],axis = 1)
test = pd.concat([test,data_delt_test],axis = 1)

data_delt_train = deltvalues(train[['gj'+'_rate'+str(i) for i in range(1,7)]])
data_delt_train = pd.DataFrame(data_delt_train,columns = ['gj_deltrate'+str(i) for i in range(1,6)])
data_delt_test = deltvalues(test[['gj'+'_rate'+str(i) for i in range(1,7)]])
data_delt_test = pd.DataFrame(data_delt_test,columns = ['gj_deltrate'+str(i) for i in range(1,6)])
train = pd.concat([train,data_delt_train],axis = 1)
test = pd.concat([test,data_delt_test],axis = 1)

############################### 训练 ##############################

le = LabelEncoder()
train['设备类型'] = le.fit_transform(train['设备类型'])
test['设备类型'] = le.transform(test['设备类型'])
train_x = train.drop(['设备id','life'],axis = 1,inplace = False)
# train_y = train['life'].apply(lambda x: math.log(x+1,2))
train_y = train['life']
test_id = test['设备id']
test = test.drop(['设备id','life'],axis = 1,inplace = False)

#分别用不同的后处理以及loss训练
#l1：loss-mae，用log（1+y）对y进行后处理
#l2: loss-rmse,不采用后处理
drop_feas1 = ['累积量参数2_min','累积量参数1_min','开关2_sum','电流信号_min',
             '部件工作时长_min','累积量参数1_ptp','累积量参数2_ptp','percent',
             '累积量参数1_t0','累积量参数2_t0','部件工作时长_t0']+['gj'+'_deltrate'+str(i) for i in range(1,6)]

drop_feas2 = ['累积量参数2_min','累积量参数1_min','开关2_sum','电流信号_min',
             '部件工作时长_min','累积量参数1_ptp','累积量参数2_ptp','percent',
             '累积量参数1_t0','累积量参数2_t0','部件工作时长_t0',
             '累积量参数1_delt','累积量参数2_delt','部件工作时长_delt',
             '累积量参数1_enum','累积量参数2_enum','部件工作时长_enum']+['gj'+'_rate'+str(i) for i in range(1,7)]

pre_train_l1,pre_test_l1,fea_imp_l1 = lgb_train_pre(train_x,train_y,test,'l1',log = True,drop_feas = drop_feas1)
pre_train_l2,pre_test_l2,fea_imp_l2 = lgb_train_pre(train_x,train_y,test,'l2',log = False,drop_feas = drop_feas2)

#训练的结果按1000为分界采用不同比例取平均
pre_test_l1 = pd.Series(pre_test_l1)
pre_test_l2 = pd.Series(pre_test_l2)
a1 = 0.6
a2 = 0.4
pre_train = pd.Series(np.zeros(pre_train_l1.shape[0]),index = pre_train_l1.index)
pre_test = pd.Series(np.zeros(pre_test_l1.shape[0]),index = pre_test_l1.index)
pre_train[pre_train_l1 < 1000] = (pre_train_l1[pre_train_l1 < 1000]*a1+pre_train_l2[pre_train_l1 < 1000]*a2)
pre_train[pre_train_l1 > 1000] = (pre_train_l1[pre_train_l1 > 1000]*a2+pre_train_l2[pre_train_l1 > 1000]*a1)
pre_test[pre_test_l1 < 1000] = (pre_test_l1[pre_test_l1 < 1000]*a1+pre_test_l2[pre_test_l1 < 1000]*a2)
pre_test[pre_test_l1 > 1000] = (pre_test_l1[pre_test_l1 > 1000]*a2+pre_test_l2[pre_test_l1 > 1000]*a1)

score1 = np.sqrt(np.square(np.log2(pre_train_l1+1)-np.log2(train_y+1)).mean())
print('score1:'+str(score1))
score2 = np.sqrt(np.square(np.log2(pre_train_l2+1)-np.log2(train_y+1)).mean())
print('score2:'+str(score2))
score = np.sqrt(np.square(np.log2(pre_train+1)-np.log2(train_y+1)).mean())
print('score:'+str(score))

#结果保存到根目录下
result = pd.DataFrame(test_id)
result['life'] = pre_test
result = result.rename(columns = {'设备id':'test_file_name'})
result_m0 = result.copy()
result['life'] = result['life'].apply(lambda x:x if x > 0 else 0)
result.to_csv(datapath+'submission.csv',header = True,index = False)