import pandas as pd
import os

def update():
    updata_data=pd.read_csv('test_a.csv')
    origin_data=pd.read_csv('user_base_info.csv')
    new_data=pd.DataFrame.merge(origin_data,updata_data,on='id',how='outer')
    new_data.loc[new_data['label_x'].isnull(),'label_x']=new_data[new_data['label_x'].isnull()]['label_y']
    new_data=new_data.drop('label_y',axis=1)
    new_data=pd.DataFrame.rename(new_data,columns={"label_x":"label"})
    new_data.to_csv("new_data.csv",index=False)
    print(new_data)
