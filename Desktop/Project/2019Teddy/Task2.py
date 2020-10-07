import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

#任务 2.1绘制生鲜类商品和一般商品每天销售金额的折线图，并分析比较两类产品的销售状况。

data = pd.read_csv('./task1_1.csv', encoding='gbk')
data.loc[:, '销售日期'] = pd.to_datetime(data.loc[:, '销售日期'].astype(str), format='%Y-%m-%d', errors='coerce')

data_new = data[['商品类型', '销售日期', '销售金额']]
#print(data_new)

#检查每个列的缺失值的占比
#print(data_new.apply(lambda x: sum(x.isnull())/len(x), axis=0))#注意0和1的区别

data_query1 = data_new.loc[:, '商品类型'] == '生鲜'
data_query2 = data_new.loc[:, '商品类型'] == '一般商品'
#print(data_query1)
data_fresh = data_new[data_query1]
data_comm = data_new.loc[data_query2]
print(data_fresh) # !!!!!这种输出的写法要学会

# 根据销售日期分组，求生鲜类商品和一般商品的每天销售金额
data_fresh_cost = data_fresh.groupby('销售日期').sum()
data_fresh_cost.columns = {'销售金额'}

data_comm_cost = data_comm.groupby('销售日期').sum()
data_comm_cost.columns = {'销售金额'}
print(data_fresh_cost)
