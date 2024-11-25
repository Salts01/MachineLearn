import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


#lista de colunas para serem normalizadas
columns_to_normalize = ['carat','cut','color','clarity','depth','table','x','y','z']


# Definindo base de dados
DB=pd.read_csv('diamonds.csv')


#Definindo valores de caracteres para numerais
cut_mapping={'Ideal':1,'Fair':2,'Very Good':3,'Good':4,'Premium':5}
DB['cut']=DB['cut'].map(cut_mapping)

color_mapping={'E':1,'D':2,'G':3,'F':4,'H':5,'J':6,'I':7}
DB['color']=DB['color'].map(color_mapping)


clarity_mapping={'SI2':1,'IF':2,'I1':3,'VVS1':4,'VVS2':5,'VS2':6,'VS1':7,'SI1':8}
DB['clarity']=DB['clarity'].map(clarity_mapping)




# Normalização
scaler = MinMaxScaler()
price_scaler = MinMaxScaler()

# Ajustar o scaler com base nos dados principais (DB)
DB[columns_to_normalize] = scaler.fit_transform(DB[columns_to_normalize])

DB[['price']] = price_scaler.fit_transform(DB[['price']])

# Separar feature e target
x = DB[columns_to_normalize]

y = DB['price']



# Separando para treino

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# Modelo de regressão Random Forest
model = RandomForestRegressor(n_estimators=15, random_state=42, min_samples_leaf=1, min_samples_split=2,max_depth=None)

# Treinando modelo
model.fit(x_train, y_train)

# Predição
predição = model.predict(x_test)

# Calculando a taxa de erro
Root_ERR=np.sqrt(metrics.mean_squared_error(predição,y_test))
Absolute_ERR=metrics.mean_absolute_error(predição,y_test)

# Exibindo taxa de Erro
print('Root_MSE: ',Root_ERR)
print('MEA: ',Absolute_ERR)




# Entrada de dados de usuário, arquivo .csv
Data_enter=input('Selecione seu arquivo: ')

if Data_enter != '' or Data_enter != 'N':

	
	# Leitura de arquivo
	Data_enter=pd.read_csv(Data_enter)

	# Realizando ajuste de colunas
	Data=pd.DataFrame(Data_enter,columns=columns_to_normalize)


	# Convertendo valores de caracteres
	Data['cut']=Data['cut'].map(cut_mapping)

	Data['clarity']=Data['clarity'].map(clarity_mapping)

	Data['color']=DB['color'].map(color_mapping)

	# Realizando normalização
	Data[columns_to_normalize]=scaler.transform(Data[columns_to_normalize])

	# Realizando treinamento
	model.fit(x,y)


	#Realizando predição 
	predict=model.predict(Data)


	# Reverter a normalização dos preços
	Origin_predict = price_scaler.inverse_transform(np.array(predict).reshape(-1, 1))




	# Exibir resultados
	for i,price in enumerate(Origin_predict.flatten(),start=1):
		print(f"Preço preditado para o diamante {i}: {price:.2f}")



