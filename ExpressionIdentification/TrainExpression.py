import tensorflow as tf
from tensorflow.keras import layers
import os

base_path = os.path.dirname(os.path.abspath(__file__))

data=os.path.join(base_path, "Data")

img_size=128
batch_size=64

epoch=200

learning_rate=0.0001

data_train=tf.keras.utils.image_dataset_from_directory( #Capturando todas as imagens dentro da pasta 'Data', a mesma já converte os valores em float32 e já inclui o valor de y automaticamente
    data  #Indica a pasta onde estará os datasets
    ,seed=123 #Indica a separação reprodutível entre treino e validação
    ,image_size=img_size #Indica o redimensionamento da imagem
    ,batch_size=batch_size #Indica a quantidade de batches
    ,label_mode='int' #Indica a classificação, no caso está em Int pois são 4 classificações. Caso fosse somente duas, a classificação deveria ser como 'binary', pois só haveria 0 e 1
    )

normalizar=layers.Rescaling(1./255) # Realizando normalização
data_train=data_train.map(lambda x,y:(normalizar(x),y))


OtimizationPC=tf.data.AUTOTUNE  # Realizando Otimização do código, para que o mesmo realize os cálculos de maneira rápida e eficiente
data_train=data_train.cache().shuffle(1000).prefetch(buffer_size=OtimizationPC)



inicializador=tf.keras.initializers.GlorotUniform() #Definição da tipagem dos pesos

Wconv1=tf.Variable(inicializador([3,3,3,64],dtype=tf.float32)) #Definição da matriz classificadora, sendo 3x3, tendo o valor do canal de entrada de 3(RGB) e 64 valores de saída
bias1=tf.Variable(inicializador([64],dtype=tf.float32))# Definindo Bias com 64 valores de saida

Wconv2=tf.Variable(inicializador([3,3,64,128],dtype=tf.float32))
bias2=tf.Variable(inicializador([128],dtype=tf.float32))

Wconv3=tf.Variable(inicializador([3,3,128,128],dtype=tf.float32))
bias3=tf.Variable(inicializador([128],dtype=tf.float32))

Wfinal=tf.Variable(inicializador([16*16*128,4],dtype=tf.float32))# Definindo Peso final com base na redução do pooling, o qual foi utilizado 3 vezes, desta forma sendo->128/2=>64/2=>32/2=>16, com 4 valores de saídas baseado na clasificação(triste, raiva, medo e feliz)
biasFinal=tf.Variable(inicializador([4],dtype=tf.float32))



def conv_function(x): #Função de convolução
    
    x=tf.nn.conv2d(x,Wconv1,strides=1,padding='SAME') # Definição do calculo de convolução, entrada (x) pelo peso(Wconv1), strides seria o quanto o classificador anda pela matriz, no caso no valor de 1 em 1 e padding seria obre o formato da borda, SAME significa que a mesma permanece
    x=tf.nn.relu(x+bias1) # Realizando operação de relu do valor pós-convolução somando os valores do bias
    x=tf.nn.max_pool2d(x,ksize=2,strides=2,padding="SAME") # Realizando processo de pooling, definindo o tamanho do pooling no ksize, onde o mesmo seria 2x2; Definindo o strides em 2.

    x=tf.nn.conv2d(x,Wconv2,strides=1,padding='SAME')
    x=tf.nn.relu(x+bias2)
    x=tf.nn.max_pool2d(x,ksize=2,strides=2,padding="SAME")

    x=tf.nn.conv2d(x,Wconv3,strides=1,padding='SAME')
    x=tf.nn.relu(x+bias3)
    x=tf.nn.max_pool2d(x,ksize=2,strides=2,padding="SAME")


    x=tf.reshape(x,[tf.shape(x)[0],16*16*128]) # Realizando processo de Flatting, onde o corre a transformação de uma matriz em apenas um vetor

    logits=tf.matmul(x,Wfinal)+biasFinal  

    return logits

def perda(logits,labels):

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))

Otimizador=tf.optimizers.Adam(learning_rate)



for epoca in range(epoch):

    loss=0

    for batch_x,batch_y in data_train:
        
        with tf.GradientTape() as tape:
            
            
            logits=conv_function(batch_x)
            perde=perda(logits,batch_y)

        gradients=tape.gradient(perde,[Wconv1,Wconv2,Wconv3,Wfinal,bias1,bias2,bias3,biasFinal]) 
        Otimizador.apply_gradients(zip(gradients,[Wconv1,Wconv2,Wconv3,Wfinal,bias1,bias2,bias3,biasFinal]))

        loss+=perde.numpy()

    print('Época: ',epoca,' Perda: ',loss/len(data_train))   

Att=tf.train.Checkpoint(Wconv1=Wconv1,Wconv2=Wconv2,Wconv3=Wconv3,Wfinal=Wfinal,bias1=bias1,bias2=bias2,bias3=bias3,biasFinal=biasFinal)

Att.write(os.path.join(base_path, "Pesos_atualizados/Pesos"))

print("Realizado treinamento com sucesso! Os pesos foram atualizados!")