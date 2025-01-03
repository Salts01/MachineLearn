from sklearn import datasets
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

#Carragendo base de dados
iris = datasets.load_iris()

#dataset com carecteristicas dos alvos
X = iris.data

#alvos
Y = iris.target

#Processo de Padronização do X
scaler_x=StandardScaler()

#Padronizando
X=scaler_x.fit_transform(X)




#Realizando separação para teste e treino
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=5)

#definições de neurônios
neuronios_entrada=4
neuronios_oculta=4
neuronios_saida=3

#Definindo Pesos
W={'oculta':tf.Variable(tf.random.normal([neuronios_entrada,neuronios_oculta]),name='w_oculto'),
   'saida':tf.Variable(tf.random.normal([neuronios_oculta,neuronios_saida]),name='w_saida')}


#Definindo Bias
bias={'oculta':tf.Variable(tf.random.normal([neuronios_oculta]),name='b_oculta'),
      'saida':tf.Variable(tf.random.normal([neuronios_saida]),name='b_saida')}

#Criando placeholders de treino
xph_train = tf.Variable(X_train,dtype=tf.float32)
yph_train = tf.Variable(Y_train,dtype=tf.int32)

#Crieando placeholders de teste
xph_test = tf.Variable(X_test,dtype=tf.float32)
yph_test = tf.Variable(Y_test,dtype=tf.int32)

#Definindo váriavel para pegar o melhor valor dos pesos. (Ação desnecessária)
atual=9.0


#Definindo função de perda
def perda(x,y):
    erro = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x,labels=y)
    return tf.reduce_mean(erro)

#Definindo otimizador
otimizador=tf.keras.optimizers.Adam(learning_rate=0.5)


#Loop de Épocas
for epoch in range(10000):

    with tf.GradientTape() as tape:

        #Realizando calculo da função soma da camada oculta
        saida_oculta=tf.add(tf.matmul(xph_train,W['oculta']),bias['oculta'])

        #Realizando processo de ativação da camada oculta
        saida_oculta_ativada=tf.nn.relu(saida_oculta)

        #Realizando processo de adição camada de saida
        saida=tf.add(tf.matmul(saida_oculta_ativada,W['saida']),bias['saida'])

        #Realizando calculo da taxa de erro
        perde=perda(saida,yph_train)

    #Realizando processo de calculo do Delta
    gradiente = tape.gradient(perde,[W['oculta'],W['saida'],bias['oculta'],bias['saida']])

    #Aplicando novos pesos
    otimizador.apply_gradients(zip(gradiente,[W['oculta'],W['saida'],bias['oculta'],bias['saida']]))

    print('Época: ',epoch,' Error: ',perde)

    
    # Processo de captura dos pesos finais
    if perde < atual:
        atual=perde
        W_final = {key:tf.identity(value) for key,value in W.items()}
        bias_final = {key:tf.identity(value) for key,value in bias.items()}

print('Menor erro: ',atual)

#realização de teste
saida_oculta=tf.add(tf.matmul(xph_test,W_final['oculta']),bias_final['oculta'])

saida_oculta_ativada=tf.nn.softmax(saida_oculta)

saida=tf.add(tf.matmul(saida_oculta_ativada,W_final['saida']),bias_final['saida'])

print('Classificação da máquina: ',tf.argmax(saida,1),"\n")

print('Resposta correta',yph_test)

yph_np=np.array(yph_test,dtype=np.int32)
saida_np=tf.argmax(saida,axis=1).numpy()

taxa_de_acerto=accuracy_score(yph_np,saida_np)*100

print('Taxa de acerto da máquina: ',taxa_de_acerto,'%')

        

    