import tensorflow as tf 
import numpy as np

#definição do banco de dados. 
X = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])

#definição do alvo
Y = np.array([[0.0],[1.0],[1.0],[0.0]])

#definindo quantidade de neurônios de cada etapa
neuronios_entrada=2
neuronios_oculta=3
neuronios_saida=1

#Realizando a definição de pesos
W ={'oculta':tf.Variable(tf.random.normal([neuronios_entrada,neuronios_oculta]),name='w_oculto'),
    'saida':tf.Variable(tf.random.normal([neuronios_oculta,neuronios_saida]),name='w_saida')}


#Definindo o BIAS
bias={'oculta':tf.Variable(tf.random.normal([neuronios_oculta]),name='b_oculta'),
      'saida':tf.Variable(tf.random.normal([neuronios_saida]),'b_saida')}

#Definindo os placeholders
xph = tf.constant(X,dtype=tf.float32,name='xph')
yph = tf.constant(Y,dtype=tf.float32,name='yph')


# Função de erro (Mean Squared Error)
def erro(y, saida):
    mse = tf.keras.losses.MeanSquaredError()  # Instanciando o MSE
    return mse(y, saida)  # Passando y e saida corretamente

# Otimizador Utilizando o Gradiente Descendente
otimizador = tf.keras.optimizers.SGD(learning_rate=0.5)

# Loop de épocas
for epoch in range(10000):
    with tf.GradientTape() as tape:
        
        #Realizando função soma da camada oculta
        saida_oculta=tf.add(tf.matmul(xph,W['oculta']),bias['oculta'])

        #Realizando ativação da função sigmoid da camada oculta
        saida_oculta_ativada=tf.sigmoid(saida_oculta)


        #Realizando função soma da camada de saída
        saida=tf.add(tf.matmul(saida_oculta_ativada,W['saida']),bias['saida'])

        #Realizando função sigmoid da camada de saída
        saida_ativada=tf.sigmoid(saida)


        #Calculando perda pelo MeanSquaredERROR
        loss = erro(yph, saida_ativada)  # Cálculo da perda

        print('Época: ',epoch,'Erro: ',loss)

# Gradientes das variáveis
    gradients = tape.gradient(loss, [W['oculta'], W['saida'], bias['oculta'], bias['saida']])

#Aplicando novos pesos
    otimizador.apply_gradients(zip(gradients, [W['oculta'], W['saida'], bias['oculta'], bias['saida']]))

#Recuperando os melhores pesos encontrados
W_final,bias_final=[W,bias]


#Realizando processo de teste com os pesos recuperados.
saida_oculta=tf.add(tf.matmul(xph,W_final['oculta']),bias_final['oculta'])
saida_oculta_ativada=tf.sigmoid(saida_oculta)

saida=tf.add(tf.matmul(saida_oculta_ativada,W_final['saida']),bias_final['saida'])
saida_ativada=tf.sigmoid(saida)

print(saida_ativada)
