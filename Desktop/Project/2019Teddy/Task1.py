"""
1. 对销售数据进行统计分析，并作可视化展示。
2. 分析顾客的消费行为。
3. 研究促销对销售的影响。
"""

#对数据作必要的预处理，在报告中列出处理步骤，将处理后的结果保存为“task1_1.csv”。
import numpy as np
import pandas as pd
import os

data = pd.read_csv("超市销售数据.csv", encoding="gbk")
#print(data.head())
data.drop(['销售月份'], axis=1, inplace=True)
print(data.shape)
print(data.dtypes)
data.loc[:, '销售日期'] = pd.to_datetime(data.loc[:, '销售日期'].astype(str), format='%Y-%m-%d', errors='coerce')
print(data.loc[:, '销售日期'])
# 检查data每列的缺失值的占比,如果有缺失值则不为0
print(data.apply(lambda x: sum(x.isnull())/len(x), axis=0))

# 删除销售日期、销售数量中的缺失值，所选列有空即删除该行
print('删除缺失值前：', data.shape)
data = data.dropna(subset=['销售日期', '销售数量'], how='any')
print('删除缺失值后：', data.shape)
print(data.apply(lambda x: sum(x.isnull())/len(x), axis=0))#0为横向，1为纵向


print(data.columns)

#统计data信息
print(data.describe().T[['mean', '50%', 'min', 'max']])#T为按项目输出数据

#删除销售数量和销售金额小于0的异常值
print('删除异常值之前：', data.shape)
data = data[(data.loc[:, '销售金额'] > 0) & (data.loc[:, '销售数量'] > 0)]
print('删除异常值之后：', data.shape)
print(data.describe().T[['mean', '50%', 'min', 'max']])
data.to_csv('./task1_1.csv', index=False, encoding='gbk')
#对数据作必要的预处理，在报告中列出处理步骤，将处理后的结果保存为“task1_1.csv”。

print(data.loc[0:5, :])#data.loc 按列提取

data_big = data[['大类名称', '销售金额']]#按行提取
print(data_big['大类名称'].value_counts(dropna=False))#因为没有0值，直接False就可以

print(data_big.head())
data_big = data_big.groupby('大类名称').sum()#按列聚合求和   .mean()为求均值
data_big.rename(columns={'销售金额': '销售金额总和'}, inplace=True)
print(data_big)

data_big.to_csv('./task1_2.csv', index=True, encoding='gbk')
#任务 1.2 统计每个大类商品的销售金额，将结果保存为“task1_2.csv”。

data_promotion = data[['中类名称', '是否促销', '销售金额']]

print(data_promotion['是否促销'].value_counts(dropna=False))#同值统计

data_promotion_yes = data_promotion.loc[data_promotion['是否促销'] == '是', :]
data_promotion_no = data_promotion.loc[data_promotion['是否促销'] == '否', :]
print(data_promotion_yes.head())

data_promotion_yes = data_promotion_yes.groupby('中类名称').sum()
data_promotion_yes.rename(columns={'销售金额': '销售金额总和'}, inplace=True)
data_promotion_no = data_promotion_no.groupby('中类名称').sum()
data_promotion_no.rename(columns={'销售金额': '销售金额总和'}, inplace=True)
# print(data_promotion_yes)
# print(data_promotion_no)

data_merge = pd.merge(data_promotion_yes, data_promotion_no, how='left', left_on='中类名称', right_on='中类名称')
data_merge.columns = list(['促销销售金额总和', '非促销销售金额总和'])
print(data_merge)
data_merge.to_csv('./task1_3.csv', index=True, encoding='gbk')
#任务 1.3统计每个中类商品的促销销售金额和非促销销售金额，将结果保存为“task1_3.csv”。

data_week_sum = data[['销售日期', '商品类型', '销售金额']]
#print(data_week_sum['销售日期'])

#计算周数
days = data_week_sum['销售日期'].astype(str).str.replace('-', '').astype(int)
weeks = ((days.max() - days.min())/7)

data_week_sum = data_week_sum.groupby('商品类型').sum()#只取商品类型一列求和显示
data_week_sum['每周销售金额'] = data_week_sum['销售金额']/weeks#没指定位置，直接append到后面

data_week_sum.drop(labels='联营商品', axis=0, inplace=True)
data_week_sum.drop(labels='销售金额', axis=1, inplace=True)#0为行标签，1为列标签
print(data_week_sum)
data_week_sum.to_csv('./task1_4.csv', index=True, encoding='gbk')
#任务 1.4统计生鲜类产品和一般产品的每周销售金额，将结果保存为“task1_4.csv”。

data_customer = data[['顾客编号', '销售日期', '销售金额']]
print(data_customer.head())


data_customer['月份'] = [x.month for x in data_customer['销售日期']]
data_customer.drop(['销售日期'], axis=1, inplace=True)

print(data_customer['月份'].value_counts(dropna=False))#看数据到底有几个月

#拆分出四个月的表
data_month1 = data_customer.loc[data_customer['月份'] == 1, :]
data_month2 = data_customer.loc[data_customer['月份'] == 2, :]
data_month3 = data_customer.loc[data_customer['月份'] == 3, :]
data_month4 = data_customer.loc[data_customer['月份'] == 4, :]
#print(data_month1)
data_month1_cost = data_month1.groupby('顾客编号').sum()
data_month1_cost.drop(['月份'], axis=1, inplace=True)
print(data_month1_cost)
data_month2_cost = data_month2.groupby('顾客编号').sum()
data_month2_cost.drop(['月份'], axis=1, inplace=True)
data_month3_cost = data_month3.groupby('顾客编号').sum()
data_month3_cost.drop(['月份'], axis=1, inplace=True)
data_month4_cost = data_month4.groupby('顾客编号').sum()
data_month4_cost.drop(['月份'], axis=1, inplace=True)

# 4个消费额表连接
data_month12_cost = pd.merge(data_month1_cost, data_month2_cost, how='left', left_on='顾客编号', right_on='顾客编号')#表连接，参数为左连接，left_on为左表合并的列，right_on同理
data_month34_cost = pd.merge(data_month3_cost, data_month4_cost, how='left', left_on='顾客编号', right_on='顾客编号')
data_month1234_cost = pd.merge(data_month12_cost, data_month34_cost, how='left', left_on='顾客编号', right_on='顾客编号')
data_month1234_cost.columns = list(['1月消费额', '2月消费额', '3月消费额', '4月消费额'])
#print(data_month1234_cost)

#任务 1.5统计每位顾客每月的消费额及消费天数，将结果保存为“task1_5.csv”，并在报告中列出用户编号为 0-10 的结果。
# 根据顾客编号列，求出每位顾客每月的消费次数
data_month1_times = data_month1['顾客编号'].value_counts(dropna=False)
data_month2_times = data_month2['顾客编号'].value_counts(dropna=False)
data_month3_times = data_month3['顾客编号'].value_counts(dropna=False)
data_month4_times = data_month4['顾客编号'].value_counts(dropna=False)
print(data_month1_times)
data_month1_times = pd.DataFrame({'顾客编号': data_month1_times.index, '次数': data_month1_times.values})
data_month2_times = pd.DataFrame({'顾客编号': data_month2_times.index, '次数': data_month2_times.values})
data_month3_times = pd.DataFrame({'顾客编号': data_month3_times.index, '次数': data_month3_times.values})
data_month4_times = pd.DataFrame({'顾客编号': data_month4_times.index, '次数': data_month4_times.values})
#print(data_month1_times)
# 4个消费次数表连接
data_month12_times = pd.merge(data_month1_times, data_month2_times, how='left', left_on='顾客编号', right_on='顾客编号')
data_month34_times = pd.merge(data_month3_times, data_month4_times, how='left', left_on='顾客编号', right_on='顾客编号')
data_month1234_times = pd.merge(data_month12_times, data_month34_times, how='left', left_on='顾客编号', right_on='顾客编号')
data_month1234_times.columns = list(['顾客编号', '1月消费次数', '2月消费次数', '3月消费次数', '4月消费次数'])
print(data_month1234_times)

#消费额表和消费次数表连接
data_cost_times = pd.merge(data_month1234_cost, data_month1234_times, how='left', left_on='顾客编号', right_on='顾客编号')
print(data_cost_times)
data_cost_times.to_csv('./task1_5.csv', index=True, encoding='gbk')


