
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import os

#import scipy.stats.mstats as mstats
creditdf = pd.read_csv('credit.csv',sep=';',encoding='cp1251')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

mis_val_percent = 100 * creditdf.isnull().sum() / len(creditdf) # - сколько процентов данных не достает
#print(mis_val_percent)
#print(creditdf['credit_count'].mode()) #-мода столбца 1
#print(creditdf['overdue_credit_count'].mode())#-мода столбца 0
creditdf.fillna({'credit_count':1,'overdue_credit_count':0},inplace=True)#Заполнение нанов модами столбцов(т.к. не достает больше 5%)
creditdf.dropna(inplace=True)#Удаление нанов в остальных столбцах(их слишком мало, поэтому можно так сделать)

creditdf.set_index('client_id',inplace=True)
creditdf['credit_sum'] = creditdf['credit_sum'].str.replace(',','.')
creditdf['score_shk'] = creditdf['score_shk'].str.replace(',','.')

#print(creditdf.select_dtypes(include=[object]).apply(pd.Series.nunique, axis = 0))


creditdf["gender"] = creditdf["gender"].astype('category')
creditdf["gender"] = creditdf["gender"].cat.codes#Кодирование столбца с гендерами

creditdf["education"] = creditdf["education"].astype('category')
creditdf["education_c"] = creditdf["education"].cat.codes#Кодирование столбца с образованием


creditdf['credit_sum']=pd.to_numeric(creditdf['credit_sum'])
creditdf['score_shk']=pd.to_numeric(creditdf['score_shk'])
creditdf['gender']=pd.to_numeric(creditdf['gender'])
correlations = creditdf.corr()['open_account_flg'].sort_values()# корелляция с целевым признаком


#print(creditdf.shape)#форма данных(количество строк, столбцов)
#print(creditdf.dtypes.value_counts())#Для просмотра количества категориальных данных
#print(creditdf.dtypes)
#print(correlations)

#sns.kdeplot(creditdf.loc[creditdf['open_account_flg'] == 0, 'age'],label='open=0')
#sns.kdeplot(creditdf.loc[creditdf['open_account_flg'] == 1, 'age'],label='open=1')
#plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#plt.legend()
ext_data = creditdf[['credit_sum', 'tariff_id', 'age', 'monthly_income', 'credit_count','overdue_credit_count','credit_month','score_shk','open_account_flg','gender','education_c']]
ext_data_corrs = ext_data.corr()
#sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
#plt.title('Correlation Heatmap');
'''
Комментарии по поводу корелляционной диаграммы:
1)На скоринговую оценку на 40% влияет номер тарифа
2)На сумму кредита на 35% влияет месячный доход
3)На сумму кредита на 23% влияет срок на который его берут
'''



#print(creditdf.describe())#-описательные статистики по датасету
#print(tabulate(ext_data.sample(10),headers='keys',tablefmt='psql'))



#НАЧИНАЕМ ОБУЧАТЬ

X=ext_data
X=X.drop(['open_account_flg'],axis=1)
y=ext_data['open_account_flg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
model_tree=DecisionTreeClassifier(max_depth=3)
model_tree.fit(X_train, y_train)
export_graphviz(model_tree, feature_names=['credit_sum', 'tariff_id', 'age', 'monthly_income', 'credit_count','overdue_credit_count','credit_month','score_shk','gender','education_c'], out_file='tree.dot',filled=True)
y_predict=model_tree.predict(X_test)
print(accuracy_score(y_test, y_predict))
#plt.show()