import pandas as pd
import numpy as np
import metric
import torch

def statics():
    res_f=open("output_a.txt",'r')
    real_label=pd.read_csv("test_a.csv")
    real_label=np.array(real_label["label"])
    # res_txt=np.array(res_f.read()).reshape(-1)
    res_txt=[]
    n=0
    correct=0
    for i in res_f:
        n+=int(i)
        res_txt.append(int(i))

    print(metric.Fscore(np.array(real_label),np.array(res_txt),1))
    print("predict label=1:::",n)
    print("real_label=1:::",np.sum(real_label))

if __name__=='__main__':
    statics()