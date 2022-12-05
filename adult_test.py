import pandas as pd
import numpy as np
import os
from sklearn

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SVMSMOTE # 过抽样处理库SMOTE

from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


# read raw data
id_test=pd.read_csv("data/test_b.csv")
id_test=list(np.asarray(id_test).reshape((-1,)))
data_t = pd.read_csv(
    "data/user_base_info.csv")
data2 = pd.read_csv(
    "data/user_his_features.csv")
user_track = pd.read_csv(
    "data/user_track.csv")
data_1 = data_t[data_t.columns[0:-1]]

user_dict=dict(user_track['id'].value_counts())
# print(user_dict)


# print("user_setage_null=",np.asarray(pd.DataFrame(data_t['user_setage']).isnull().any(axis=1) == 1))
# print("label=1:",np.array(pd.DataFrame(data_t['label'])==1).reshape((-1,)))
# print(np.array(pd.DataFrame(data_t['label'])==1)
#                         &
#                     np.array(pd.DataFrame(data_t['user_setage']).isnull().any(axis=1) == 1).reshape((-1,)))


# for i in user_dict.keys():
#     data_t.loc[data_t['id']==i,'times']=user_dict[i]
#     data_t.loc[data_t['id']==i,'weenkend_rate']=np.sum(user_track.loc[user_track['id'] == i,'is_weekend'])/user_dict[i]
#     data_t.loc[data_t['id']==i,'avg_period']=np.sum(user_track.loc[user_track['id'] == i,'last_hour']-user_track.loc[user_track['id'] == i,'early_hour'])/user_dict[i]
#     data_t.loc[data_t['id'] == i, 'avg_start'] = np.sum(user_track.loc[user_track['id'] == i, 'early_hour'] ) / user_dict[i]
#     data_t.loc[data_t['id'] == i, 'avg_end'] = np.sum(user_track.loc[user_track['id'] == i, 'last_hour']) / user_dict[i]
    #data_t.loc[data_t['id'] == i, ''] = np.sum( user_track.loc[user_track['id'] == i, 'is_weekend'] )/user_dict[i]

print(data_t.columns)
#static the distribution of data
for col in data_t.columns:
    print(
        "{} 指标的缺失样本数 = {:d},其中标签为1的个数={:d}".format(
            col, np.sum(
                pd.DataFrame(
                    data_t[col]).isnull().any(
                    axis=1) == 1),
                np.sum( np.array(pd.DataFrame(data_t['label'])==1).reshape((-1,))
                        &
                    np.array(pd.DataFrame(data_t[col]).isnull().any(axis=1) == 1))
        )
    )


# print( data_1 )
# print(data2)
# print(label)


# handle missing value
# data1_isNA = data_1.isnull()
# data2_isNA = data2.isnull()
# label_isNA = label.isnull()
# print(data1_isNA.any(axis=1))
# print(data2_isNA.any(axis=1))
# print(label_isNA.any(axis=1))


to_del=[]


for index, row in data_t.iterrows():
#     print("{} 样本的缺失指标数 = {:d}".format(
#        int(row['id']), np.sum(data_t.iloc[index].isnull() == 1)))

    if np.sum(data_t.iloc[index].isnull() == 1)>=5 or np.isnan(data_t.loc[index,'label'])==1:
        to_del.append(index)


# print(to_del)

#copy origin
data_org=data_t.drop('label',axis=1)
data_org2=data2


#create test file
data_test=pd.merge(data_org,data_org2,on='id')
# print(data_test)
# print(data_test['id'].isin(id_test))
# print(id_test)
data_test=data_test.loc[data_test['id'].isin(id_test),:]


#correlation analysis
Pcor_matrix=data_t.corr(method="pearson")
Scor_matrix=data_t.corr(method="spearman")
# # sns.pairplot(data_t)
sns.heatmap(Pcor_matrix)
plt.savefig("Pcorr")
plt.clf()
sns.heatmap(Scor_matrix)
plt.savefig("Scorr")
for feature in data_t.columns:
    plt.hist(data_t.loc[data_t['label'] == 1, feature] ,density=True, color="#FF0000", bins='auto', alpha=.9)
    plt.hist(data_t.loc[data_t['label'] == 0, feature] ,density=True, color="#C1F320", bins='auto', alpha=.5)
    #sns.jointplot(x = feature, y = 'label', data = data_t ,kind = 'scatter',hue='app_version')
    plt.savefig("{}_label".format(feature))
    plt.clf()
if 0:
    plt.show()


data_test=data_test.drop( "utm_channel", axis=1 )
# data_test=data_test.drop( "user_setage", axis=1 )
# data_test=data_test.drop( "playcard_level", axis=1 )
data_test=data_test.drop( "signup_day", axis=1 )
data_test=data_test.drop( "app_version", axis=1 )


#KNN impute
print("origin data_test:\n",data_test)
# data_test=data_test.fillna(data_test.mean())
imputer1=KNNImputer(weights='distance')
# print(list(data_test.columns))
data_test_cols=list(data_test.columns)
data_test_cols.remove('user_setage')
data_test_cols.remove('id')
# print(data_test)
# print("KNN_dealed:\n",pd.DataFrame(imputer1.fit_transform(data_test.drop(['id','user_setage'],axis=1)),columns=data_test_cols))
# print("data_test.loc[:,['id','user_setage']]:\n",data_test.loc[:,['id','user_setage']])
data_test=pd.concat([pd.DataFrame(imputer1.fit_transform(data_test.drop(['id','user_setage'],axis=1)),columns=data_test_cols).reset_index(drop=True),data_test.loc[:,['id','user_setage']].reset_index(drop=True)],axis=1)
#data_test=data_test.sort_values(by="id")
print("after KNN:\n",data_test)

#Random forest impute
# rf_train=data_test.loc[data_test['user_setage'].notnull(),:]
# rf_test=data_test.loc[data_test['user_setage'].isnull(),:]
# x_rf_train=rf_train[data_test_cols]
# y_rf_train=rf_train['user_setage']
# x_rf_test=rf_test[data_test_cols]
# rf_imputor=RandomForestRegressor(n_estimators=100)
# rf_imputor=rf_imputor.fit(x_rf_train,y_rf_train)
# y_pred=rf_imputor.predict(x_rf_test)
# rf_test['user_setage']=y_pred
# print("data_test['id']\n",data_test['id'])
# print("rf_train\n",rf_train)
# print("rf_test\n",rf_test)
# data_test=pd.concat([rf_train.reset_index(drop=True),rf_test.reset_index(drop=True)])

# the order of test data
id_test_cat=pd.api.types.CategoricalDtype(categories=id_test,ordered=True)
data_test['id']=data_test['id'].astype(id_test_cat)
data_test.sort_values('id',inplace=True)
print("after random forest:\n",data_test)


#del useless col
data_t=data_t.drop( "utm_channel", axis=1 )
# data_t=data_t.drop( "user_setage", axis=1 )
# data_t=data_t.drop( "playcard_level", axis=1 )
data_t=data_t.drop( "signup_day", axis=1 )
data_t=data_t.drop( "app_version", axis=1 )
# print("在id_test中而不在to_del中:::",len(list(set(id_test).difference(set(data_t.loc[to_del,'id'])))))
# print("__________data_t_before____________\n",data_t)
# print("data_t[data_t['id'].isin(id_test)].index===============\n",data_t[data_t['id'].isin(id_test)].index)
# print("data_t['id'].isin(id_test).any()===============\n",data_t['id'].isin(id_test).any())
# print("data[data_t['id'].isin(id_test)]===============\n",data_t[data_t['id'].isin(id_test)])

data_t=data_t.drop(index=to_del)
# print("__________data_t_middle____________\n",data_t)
# print("data_t[data_t['id'].isin(id_test)].index===============\n",data_t[data_t['id'].isin(id_test)].index)
# print("data_t['id'].isin(id_test).any()===============\n",data_t['id'].isin(id_test).any())
# print("data[data_t['id'].isin(id_test)]===============\n",data_t[data_t['id'].isin(id_test)])
data_t=data_t.drop(index=data_t[data_t['id'].isin(id_test)].index)
# print("__________data_t_after____________\n",data_t)

data_1 = data_t.drop('label',axis=1)

data2=data2.drop(data2[data2['id'].isin(id_test)].index)

# print(data2)


data_input=pd.merge(data_1,data2,on='id')
#get label
label = data_t.loc[:,['id','label']]
print(label)

data_all=pd.merge(data_input,label,on='id')
# data_input=data_all.drop("id",axis=1)
data_input=data_all.drop("label",axis=1)
label=data_all.loc[:,['label']]


print("origin data_input:\n",data_input)
#KNN fillna
imputer= KNNImputer(weights='distance')
data_input_cols=list(data_input.columns)
data_input_cols.remove('id')
data_input_cols.remove('user_setage')
data_input=pd.concat([pd.DataFrame(imputer.fit_transform(data_input.drop(['id','user_setage'],axis=1)),columns=data_input_cols).reset_index(drop=True),data_input[['id','user_setage']].reset_index(drop=True)],axis=1)
# data_input=pd.DataFrame(imputer.fit_transform(data_input),columns=data_input_cols)
# data2=data2.drop(index=to_del) not need for merge will clean the diff between data_1 and data2 automatically
# data2=data2.fillna(data2.mean())

print("after KNN:\n",data_input)
#Random forest impute
# rf_train=data_input.loc[data_input['user_setage'].notnull(),:]
# rf_test=data_input.loc[data_input['user_setage'].isnull(),:]
# x_rf_train=rf_train[data_input_cols]
# y_rf_train=rf_train['user_setage']
# x_rf_test=rf_test[data_input_cols]
# rf_imputor=RandomForestRegressor(n_estimators=100)
# rf_imputor=rf_imputor.fit(x_rf_train,y_rf_train)
# y_pred=rf_imputor.predict(x_rf_test)
# rf_test['user_setage']=y_pred
# data_input=pd.concat([rf_train.reset_index(drop=True),rf_test.reset_index(drop=True)])
# print(data_input.columns)
# print("after random forest:\n",data_input)
data_input=data_input.fillna(value=data_input.mean())


#oversampling
model_smote=SMOTE()
model_adasyn=ADASYN()
model_svm=SVMSMOTE()
model_undersample = RandomUnderSampler() # 建立随机欠采样模型对象
data_input,label=model_svm.fit_resample(data_input,label)
pd.concat([data_input,label],axis=1).to_csv('data/v1/all_info.csv',index=False)
# print("label里的标签1个数=",np.sum(label==1))
# print("label里的标签0个数=",np.sum(label==0))
# print(data_input)
# print(label)

column=['id','real_age', 'gender', 'playcard_level', 'playcard_point',
       'playcard_coupon_num', 'is_push_open', 'manufacturer', 'model', 'times',
       'weenkend_rate', 'avg_period', 'avg_start', 'avg_end', 'add_all_num',
       'add_under_18_num', 'add_under_30_num', 'add_under_setage_num',
       'view_all_num', 'view_under_18_num', 'view_under_30_num',
       'view_under_setage_num', 'msg_all_num', 'msg_under_18_num',
       'msg_under_30_num', 'msg_under_setage_num', 'gift_all_num',
       'gift_under_18_num', 'gift_under_30_num', 'gift_under_setage_num',
       'user_setage']
data_input=data_input[column]
data_test=data_test[column]
# data_input=data_input.astype('int')
# data_test=data_test.astype('int')
xtrain,xtest,ytrain,ytest=train_test_split(data_input,label,test_size=0.3)
pd.concat([xtrain,ytrain],axis=1).to_csv('data/v1/train.csv',index=False)
pd.concat([xtest,ytest],axis=1).to_csv('data/v1/valid.csv',index=False)
data_test.to_csv('data/v1/test_b.csv',index=False)


# rfc=RandomForestClassifier(random_state=0,n_estimators=100)
# xgb=xgboost.XGBModel(object='binary:logistic',eval_metric='auc')
# xgb.fit(xtrain,ytrain)
# rfc=RandomForestClassifier(random_state=0,n_estimators=100,class_weight={0:1,1:np.sum(label==1)/np.sum(label==0)})
# rfc=rfc.fit(xtrain,ytrain)
# #evaluate random forest
# score_r=rfc.score(xtest,ytest)
# print("random froest:{}".format(score_r))
# #predict

#define scoring fun
def my_score(y_true,y_pred):
    return np.sum(y_pred)

# score=make_scorer(my_score)

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


# para_grid={'max_depth':list(range(5,6)),'scale_pos_weight':list(np.asarray(range(1,4))*0.1)}#'eta':list(np.asarray(range(1,20))*0.01)
# grid_xgb=GridSearchCV(xgb,para_grid,scoring="accuracy",refit=True)
# def training_f():
# grid_xgb.fit(xtrain,ytrain)

# print(grid_xgb.cv_results_)
# plot_grid_search(grid_xgb.cv_results_,list(np.asarray(range(1,4))*0.1),list(range(5,6)),'scale_pos_weight','Max_depth')
if 0:
    plt.show()

# grid_rfc=GridSearchCV(rfc,para_grid,scoring='f1')
# grid_rfc.fit(data_input,label)
#
#
# pd.DataFrame(grid_rfc.cv_results_).to_excel('para_choose.xls')


# print(data_test.columns)
# print(data_test.isna().any())
# data_test=data_test.drop('id',axis=1)
# res=xgb.predict(data_test)
# res_f=open("C:\\Users\\86133\\Desktop\\2021微派种子杯赛题\\res.txt",'w')
# for i in res:
#     print(round(i),file=res_f)
# print("predict label=1:::",np.sum(np.around(res)))

