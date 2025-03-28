from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

dados=pd.read_csv('advertising.csv')

x=dados[['TV','Radio','Newspaper']]
y=dados['Sales']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)

RL = LinearRegression()

RL.fit(x_train,y_train)

print('a cada 1 dolar investido temos o coeficiente de vendas aumentadas em:',list(zip(['TV','Radio','Newspaper'],RL.coef_)))

y_prev=RL.predict(x_test)

ERR_tax=np.sqrt(metrics.mean_squared_error(y_test,y_prev))

print('taxa de erro: ',ERR_tax)



