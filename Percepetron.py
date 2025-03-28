import tensorflow as tf
import numpy as np

X = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])
Y = np.array([[0.0],[0.0],[0.0],[1.0]])

#Definição da váriavel de pesos
W = tf.Variable(tf.zeros([2,1],dtype=tf.float64))

#Definição da função de ativação
def step(x):
    return tf.cast(tf.math.greater_equal(x,1),tf.float64)

for i in range(1,16):
    #Realizando calculo dos pesos pelas entradas

    saida=tf.matmul(X,W)

    #Realizando ativação
    saida_ativada=step(saida)

    erro=tf.subtract(Y,saida_ativada)

    delta=tf.matmul(X,erro,transpose_a=True)

    treinamento = W.assign_add(tf.multiply(delta,0.1))

    erro_total,_ =[erro,treinamento]

    erro_soma=tf.reduce_sum(erro_total)

    print('Época: ',i,' Erro:',erro_soma)


    if erro_soma == 0:
        break

