import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist 

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train/255.0
x_test=x_test/255.0

x_train=x_train[...,tf.newaxis]#Realiza a adiciona mais uma coluna como canal de entrada, neste caso o valor seria 1, sendo a escala de cinza
x_test=x_test[...,tf.newaxis]


xph_train=tf.Variable(x_train,dtype=tf.float32)
xph_test=tf.Variable(x_test,dtype=tf.float32)

yph_train=tf.Variable(y_train,dtype=tf.int32)
yph_test=tf.Variable(y_test,dtype=tf.int32)

epochs=5
batchs=256
learning_rate=0.001

inicializador=tf.keras.initializers.GlorotUniform()

W_conv1=tf.Variable(inicializador([3,3,1,32],dtype=tf.float32))#Criando um peso classificador 3x3, esperando um canal de entrada e 32 de saída
bias_conv1=tf.Variable(inicializador([32],dtype=tf.float32))

W_conv2=tf.Variable(inicializador([3,3,32,64],dtype=tf.float32))#Criando um peso classificador 3x3,esperando 32 canais de entrada e gerando 64 de saída
bias_conv2=tf.Variable(inicializador([64],dtype=tf.float32))

# Ao chegar na criação do ultimo peso, deve considerar o tamanho atual da matriz depois de dois pooligns, ou seja: pooling 1-> 28x28(tamanho original) pelo 2x2 é igual a 14x14; pooling 2-> 14x14(tamanho depois do pooling) pelo 2x2 é igual a 7x7;
W_denso=tf.Variable(inicializador([7*7*64,10],dtype=tf.float32))#Criando o ultimo peso para a camada densa, onde esperaria 3.136 para a saída de 10 resultados para classificação da máquina
bias_denso=tf.Variable(inicializador([10],dtype=tf.float32))

train_datasets=tf.data.Dataset.from_tensor_slices((xph_train,yph_train)).shuffle(60000).batch(batchs)# Realizando a separação em batches

def conv_function(x):
    
    x=tf.nn.conv2d(x,W_conv1,strides=1,padding='SAME')
    x=tf.nn.relu(x+bias_conv1)
    x=tf.nn.max_pool2d(x,2,2,'SAME')

    x=tf.nn.conv2d(x,W_conv2,1,'SAME')
    x=tf.nn.relu(x+bias_conv2)
    x=tf.nn.max_pool2d(x,2,2,'SAME')

    x=tf.reshape(x,[-1,7*7*64])

    logits=tf.matmul(x,W_denso)+bias_denso

    return logits

def perda(logits,labels):

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))

otimizador=tf.optimizers.Adam(learning_rate)


for epoch in range(epochs):
    
    loss=0
    
    for batch_x,batch_y in train_datasets:
        
        with tf.GradientTape() as tape:
        
            logits=conv_function(batch_x)
            perde=perda(logits,batch_y)

        gradients=tape.gradient(perde,[W_conv1,W_conv2,W_denso,bias_conv1,bias_conv2,bias_denso])
        otimizador.apply_gradients(zip(gradients,[W_conv1,W_conv2,W_denso,bias_conv1,bias_conv2,bias_denso]))

        loss+=perde.numpy()

    print('Época: ',epoch+1,'perda: ',loss/len(train_datasets))    






