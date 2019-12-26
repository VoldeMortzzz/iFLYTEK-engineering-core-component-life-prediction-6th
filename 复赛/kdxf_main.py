import numpy as np
import pandas as pd
import os
import math
import gc
import pickle
from matplotlib.colors import LogNorm
from pylab import *
import seaborn as sns
from multiprocessing import Pool
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Normalizer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE
import warnings 
import lightgbm as lgb
# import xgboost as xgb
warnings.filterwarnings('ignore')
datapath = '/media/ant2018/kdxf/fusai/'#根目录

################################### 函数function #################################

#温度分段统计特征1（mean、std、总errornum）
def create_errornum_wendu_single(file,dic_file,perlist,col):
    n = 10
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/n) for i in range(1,n+1)]
        inds = [0] + inds
        num1 = 0
        num2 = 0
        for i in range(n):
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

#温度分段统计特征2（sc、std、errornum）
def create_errornum_wendu_single2(file,dic_file,perlist,col):
    n = 10
    sam = pd.read_csv(file)
    life_max = sam['部件工作时长'].max()
    csv_name = file.split('/')[-1]
    per_indexes = {}
    for per in perlist:
        dic_file[csv_name+'_'+str(per)] = []
        index = np.max(sam[sam['部件工作时长'] <= life_max*per].index.to_list())
        inds = [math.ceil(i*index/n) for i in range(1,n+1)]
        inds = [0] + inds
        num1 = 0
        num2 = 0
        for i in range(n):
            start = inds[i]
            end = inds[i+1]
            data = sam.loc[start:end,col]
            mean = data.mean()
            std = data.std()
            sc = data.skew()
            dic_file[csv_name+'_'+str(per)].append(sc)
#             ku = data.kurt()
#             dic_file[csv_name+'_'+str(per)].append(ku)
#             shapefacter = math.sqrt(pow(data.mean(),2) + pow(data.std(),2))/(abs(data).mean()) 
#             dic_file[csv_name+'_'+str(per)].append(shapefacter)
#             peekfacter = (max(data)) / math.sqrt(pow(data.mean(),2) + pow(data.std(),2))
#             dic_file[csv_name+'_'+str(per)].append(peekfacter)
#             try:
#                 num1 = (sam.loc[start:end,col] >= mean + std*1.8).value_counts()[True]
#             except:
#                 num1 = 0
            try:
                num2 = (sam.loc[start:end,col] <= mean - std*1.8).value_counts()[True]
            except:
                num2 = 0
#             dic_file[csv_name+'_'+str(per)].append(num1)
            dic_file[csv_name+'_'+str(per)].append(num2)

    return dic_file

#压力分段统计特征
def create_errornum_yali_single(file,dic_file,perlist,col):
    n = 10
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
        inds = [math.ceil(i*index/n) for i in range(1,n+1)]
        inds = [0] + inds
        num2 = 0
        for i in range(n):
            start = inds[i]
            end = inds[i+1]
#             mean = sam.loc[start:end,col].mean()
#             dic_file[csv_name+'_'+str(per)].append(mean)
            std = sam.loc[start:end,col].std()
            dic_file[csv_name+'_'+str(per)].append(std)
        dic_file[csv_name+'_'+str(per)].append(mode)
    return dic_file

#together1函数用于多线程处理上述特征
def get_together1(n_cpu,filelist,func,data_type,perlist,col):
    
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
def dic_merge_errornum1(dic,data,col):
    cols = []
    n = 10
    for i in range(1,n+1):
        cols.append(col+'_mean'+str(i))
        cols.append(col+'_std'+str(i))
    values = pd.DataFrame(dic.values(),columns = cols+[col+'n_err+',col+'n_err-'])
    values_c = values.copy()
    values[col+'mean_std'] = values_c[[col+'_mean'+str(i) for i in range(1,n+1)]].std(axis = 1)
    values[col+'_errsum'] = values_c[[col+'n_err+',col+'n_err-']].sum(axis = 1)
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    del data[col+'n_err-']
    
    return data

def dic_merge_errornum1_plus(dic,data,col):
    cols = []
    n = 10
    for i in range(1,n+1):
        cols.append(col+'_sc'+str(i))
#         cols.append(col+'_ku'+str(i))
#         cols.append(col+'_shapefacter'+str(i))
#         cols.append(col+'_peekfacter'+str(i))
#         cols.append(col+'_n+'+str(i))
        cols.append(col+'_n-'+str(i))
    values = pd.DataFrame(dic.values(),columns = cols)
    values_c = values.copy()
    values[col+'sc_mean'] = values_c[[col+'_sc'+str(i) for i in range(1,n+1)]].mean(axis = 1)
#     values[col+'ku_mean'] = values_c[[col+'_ku'+str(i) for i in range(1,n+1)]].mean(axis = 1)
#     values[col+'shapefacter_mean'] = values_c[[col+'_shapefacter'+str(i) for i in range(1,n+1)]].mean(axis = 1)
#     values[col+'peekfacter_mean'] = values_c[[col+'_peekfacter'+str(i) for i in range(1,n+1)]].mean(axis = 1)
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    
    return data

def dic_merge_errornum2(dic,data,col):
    cols = []
    n = 10
    for i in range(1,n+1):
#         cols.append(col+'_mean'+str(i))
        cols.append(col+'_std'+str(i))
    values = pd.DataFrame(dic.values(),columns = cols+[col+'_mode'])
    values_c = values.copy()
#     values[col+'mean_std'] = values_c[[col+'_mean'+str(i) for i in range(1,5)]].std(axis = 1)#5
    values[col+'std_mean'] = values_c[[col+'_std'+str(i) for i in range(1,5)]].mean(axis = 1)
    values['设备id'] = dic.keys()
    values['percent'] = values['设备id'].apply(lambda x : x.split('_')[1])
    values['设备id'] = values['设备id'].apply(lambda x : x.split('_')[0])
    data = pd.merge(data,values,on = ['设备id','percent'],how = 'left')
    
    return data

#pca、tsne降维，用于降维分段统计特征去除噪声
def runSvd(data_train,data_test,fea,n):
    normalizer = Normalizer(copy = False)
    data = pd.concat([data_train,data_test],axis = 0,ignore_index = True)
    data = normalizer.fit_transform(data)
    svd = TruncatedSVD(n_components = n)
    svd.fit(data)
    data_train = normalizer.transform(data_train)
    data_train = svd.transform(data_train)
    data_test = normalizer.transform(data_test)
    data_test = svd.transform(data_test)
    data_train = pd.DataFrame(data_train,columns = [fea+'_'+str(i) for i in range(n)])
    data_test = pd.DataFrame(data_test,columns = [fea+'_'+str(i) for i in range(n)])
    return data_train,data_test

def runTsne(data_train,data_test,fea,n):
    normalizer = Normalizer(copy = False)
    data = pd.concat([data_train,data_test],axis = 0,ignore_index = True)
    data = normalizer.fit_transform(data)
    tsne = TSNE(n_components = n,init='pca',perplexity=30)
    data = tsne.fit_transform(data)
    data = pd.DataFrame(data,columns = [fea+'_'+str(i) for i in range(n)])
    data_train = data.loc[:train.shape[0]-1,:]
    data_test = data.loc[train.shape[0]:,:].reset_index(drop = True)
    return data_train,data_test

#lgb训练函数
def lgb_train_pre(train_x,train_y,test,metric,log,drop_feas,weight_flag,sample_weight,drop_seq,save_model,seed):
    
#     drop_seq = []#train_y[train_y>6000].index
    train_x = train_x.drop(drop_feas,axis = 1)
    test = test.drop(drop_feas,axis = 1)
    kf = model_selection.KFold(n_splits = 5,random_state = seed,shuffle = True)
    paras_lgb = {'boosting_type':'gbdt',
                 'objective':metric,#'regression_'+metric ,
                 'random_state':2019,
                 'num_leaves':41,
                 'learning_rate':0.01,
                 'n_estimators':50000,#40000
                 'max_depth': -1, #6
                 'feature_fraction':0.8,
                 'bagging_fraction':0.7,
                 'bagging_freq':2,
#                  'min_child_samples': 30,
#                  'min_child_weight': 0.001,
                 'reg_alpha':0, 
                 'reg_lambda':2}

    lgbr = lgb.LGBMRegressor(**paras_lgb)
    categorical_feature = []#'设备类型']#,'time_isminus']
    pre_train = pd.Series(index = train_y.index)
    pre_test = []
#     eval_m = {'mae':'mae',my_loss:'rmse'}
    if log:
        train_y = train_y.apply(lambda x: math.log(x+1,2))
    
    for i,(train_index,test_index) in enumerate(kf.split(train_x)):
        train_index = list(train_index)
        for ind in train_index:
            if ind in drop_seq:
                train_index.remove(ind)
        if weight_flag:
            weights = sample_weight[train_index] 
        else:
            weights = None       
        x_train = train_x.loc[train_index,:]
        y_train = train_y[train_index]
        x_test = train_x.loc[test_index,:]
        y_test = train_y[test_index]
        lgbr.fit(x_train,y_train,eval_set = [(x_test,y_test)],eval_metric = 'rmse',early_stopping_rounds = 300,
                 sample_weight = weights,verbose = False,categorical_feature = categorical_feature)
        if save_model:
            with open(datapath+'model_{}.pkl'.format(metric+'_'+str(i)),'wb') as f:
                pickle.dump(lgbr,f)
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
test_files = os.listdir(datapath+'test2_new')
#文件分隔比例
# perlist_train = [0.45,0.55,0.63,0.75,0.85]
perlist_train = [0.50,0.54,0.58,0.65,0.72,0.77]
perlist_test = [1]

#温度
dic_wendu_train = get_together1(12,train_files,create_errornum_wendu_single,'train',perlist_train,'温度信号')
dic_wendu_test = get_together1(12,test_files,create_errornum_wendu_single,'test2',perlist_test,'温度信号')

dic_wendu_train2 = get_together1(12,train_files,create_errornum_wendu_single2,'train',perlist_train,'温度信号')
dic_wendu_test2 = get_together1(12,test_files,create_errornum_wendu_single2,'test2',perlist_test,'温度信号')

#压力信号2
dic_yali2_train = get_together1(12,train_files,create_errornum_yali_single,'train',perlist_train,'压力信号2')
dic_yali2_test = get_together1(12,test_files,create_errornum_yali_single,'test2',perlist_test,'压力信号2')
print('分段统计特征计算完成...')

############################## 特征工程2:拼接各特征dict到train/test ###############################

train = pd.read_csv(datapath+'train.csv',header = 0)
test = pd.read_csv(datapath+'test.csv',header = 0)
train['percent'] = train['percent'].apply(str)
test['percent'] = test['percent'].apply(str)

#去掉异常样本
drop_sam = ['558df5deb7c15a6b8665.csv',
            '69d0c0d1b3a2308fa7fe.csv',
            '27b553e8ea6b0c30c2ad.csv']
#             'b0970387350723d3bd59.csv']
train = train[(1-train['设备id'].isin(drop_sam)).astype(bool)].reset_index(drop = True)

#温度
train = dic_merge_errornum1(dic_wendu_train,train,'wendu')
test = dic_merge_errornum1(dic_wendu_test,test,'wendu')

train = dic_merge_errornum1_plus(dic_wendu_train2,train,'wendu')
test = dic_merge_errornum1_plus(dic_wendu_test2,test,'wendu')

#压力信号2
train = dic_merge_errornum2(dic_yali2_train,train,'yali2')
test = dic_merge_errornum2(dic_yali2_test,test,'yali2')

#温度分段特征pca降维后效果更好
cols_wd1 = ['wendu_mean'+str(i) for i in range(1,11)]+['wendu_std'+str(i) for i in range(1,11)]+\
            ['wendun_err+', 'wendumean_std', 'wendu_errsum']

# cols_wd2 = []
# for fea in ['sc','ku','shapefacter','peekfacter','n-','n+']:
#     cols_wd = cols_wd+['wendu_'+fea+str(i) for i in range(1,11)]+['wendu'+fea+'_mean']

pca_wd1 = ['wendu_sc'+str(i) for i in range(1,11)]+['wendusc_mean']
pca_wd2 = ['wendu_n-'+str(i) for i in range(1,11)]
cols_wd2 = pca_wd1+pca_wd2

train[cols_wd1] = train[cols_wd1].fillna(train[cols_wd1].mean(axis = 0))
test[cols_wd1] = test[cols_wd1].fillna(test[cols_wd1].mean(axis = 0))
train_wd0,test_wd0 = runSvd(train[cols_wd1],test[cols_wd1],'wendu0',int(len(cols_wd1)*0.35))

train[pca_wd1] = train[pca_wd1].fillna(train[pca_wd1].mean(axis = 0))
test[pca_wd1] = test[pca_wd1].fillna(test[pca_wd1].mean(axis = 0))
train_wd1,test_wd1 = runSvd(train[pca_wd1],test[pca_wd1],'wendu1',int(len(pca_wd1)*0.35))

train[pca_wd2] = train[pca_wd2].fillna(train[pca_wd2].mean(axis = 0))
test[pca_wd2] = test[pca_wd2].fillna(test[pca_wd2].mean(axis = 0))
train_wd2,test_wd2 = runSvd(train[pca_wd2],test[pca_wd2],'wendu2',int(len(pca_wd2)*0.35))

train = pd.concat([train,train_wd0,train_wd1,train_wd2],axis = 1)
test = pd.concat([test,test_wd0,test_wd1,test_wd2],axis = 1)
print('pca降维完成...')

############################### 训练 ##############################

le = LabelEncoder()
train['设备类型'] = le.fit_transform(train['设备类型'])
test['设备类型'] = le.transform(test['设备类型'])
train_x = train.drop(['设备id','life'],axis = 1,inplace = False)
train_y = train['life']
test_id = test['设备id']
test_x = test.drop(['设备id','life'],axis = 1,inplace = False)

#特征选择；
cols_yl2 = ['yali2_std'+str(i) for i in range(1,11)]#yali2std_mean,yali2_mode

cols = []
for fea in ['电流信号','流量信号','温度信号','压力信号2','压力信号1','转速信号2','转速信号1']:
    for x in ['sc','ku','shapefacter','peekfacter']:
        cols.append(fea+'_'+x)
cols.remove('电流信号_peekfacter')
cols.remove('流量信号_peekfacter')
cols.remove('流量信号_ku')
cols.remove('温度信号_shapefacter')

drop_feas = ['开关2_sum','电流信号_min','percent','设备类型','部件工作时长_min','部件工作时长_mean','部件工作时长_ptp',
             '累积量参数1_min','累积量参数1_mean', '累积量参数1_ptp','累积量参数2_min',
             '累积量参数2_mean','累积量参数2_ptp']+cols_wd2+cols_yl2+cols+cols_wd1

#分别用不同的后处理以及loss训练；
#采用bagging的方式，用2019、2020、2021作为kfold的seed，结果取平均
#l1：loss-mae，用log（1+y）对y进行后处理
#l2: loss-rmse,不采用后处理
pre_train_l1_list = []
pre_train_l2_list = []
pre_test_l1_list = []
pre_test_l2_list = []

print('开始训练...')
for i in range(3):
    pre_train_l1i,pre_test_l1i,fea_imp_l1 = lgb_train_pre(train_x,train_y,test_x,'regression_l1',log = True,drop_feas = drop_feas,
                                                        weight_flag = False,sample_weight = [],drop_seq = [],save_model = False,seed = 2019+i)
    # pre_train_l2i,pre_test_l2i,fea_imp_l2 = lgb_train_pre(train_x,train_y,test_x,'regression_l2',log = False,drop_feas = drop_feas,
    #                                                     weight_flag = False,sample_weight = [],drop_seq = [],save_model = False,seed = 2019+i)
    pre_train_l1_list.append(pre_train_l1i)
    pre_test_l1_list.append(pre_test_l1i)
    # pre_train_l2_list.append(pre_train_l2i)
    # pre_test_l2_list.append(pre_test_l2i)

pre_train_l1 = np.array(pre_train_l1_list).mean(axis = 0)
pre_test_l1 = np.array(pre_test_l1_list).mean(axis = 0)
# pre_train_l2 = np.array(pre_train_l2_list).mean(axis = 0)
# pre_test_l2 = np.array(pre_test_l2_list).mean(axis = 0)
pre_train_l1 = pd.Series(pre_train_l1)
# pre_train_l2 = pd.Series(pre_train_l2)

print('训练结束！')


# #训练的结果按3.5k为分界采用不同比例取平均（复赛仅使用了l1的线上结果更好）
pre_test_l1 = pd.Series(pre_test_l1)
# pre_test_l2 = pd.Series(pre_test_l2)
# a1 = 0.8
# a2 = 0.2
# hold = 3500
# pre_train = pd.Series(np.zeros(pre_train_l1.shape[0]),index = pre_train_l1.index)
# pre_test = pd.Series(np.zeros(pre_test_l1.shape[0]),index = pre_test_l1.index)
# pre_train[pre_train_l2 < hold] = (pre_train_l1[pre_train_l2 < hold]*a1+pre_train_l2[pre_train_l2 < hold]*a2)
# pre_train[pre_train_l2 > hold] = (pre_train_l1[pre_train_l2 > hold]*a2+pre_train_l2[pre_train_l2 > hold]*a1)
# pre_test[pre_test_l2 < hold] = (pre_test_l1[pre_test_l2 < hold]*a1+pre_test_l2[pre_test_l2 < hold]*a2)
# pre_test[pre_test_l2 > hold] = (pre_test_l1[pre_test_l2 > hold]*a2+pre_test_l2[pre_test_l2 > hold]*a1)
score1 = np.sqrt(np.square(np.log(pre_train_l1+1)-np.log(train_y+1)).mean())
print('score1:'+str(score1))
# score2 = np.sqrt(np.square(np.log(pre_train_l2+1)-np.log(train_y+1)).mean())
# print('score2:'+str(score2))
# score = np.sqrt(np.square(np.log(pre_train+1)-np.log(train_y+1)).mean())
# print('score:'+str(score))

#结果保存到根目录下
result = pd.DataFrame(test_id)
result['life'] = pre_test_l1
result = result.rename(columns = {'设备id':'test_file_name'})
result_m0 = result.copy()
result['life'] = result['life'].apply(lambda x:x if x > 0 else 0)
result.to_csv(datapath+'submission.csv',header = True,index = False)
print('结果submission.csv已保存到根目录：'+datapath)