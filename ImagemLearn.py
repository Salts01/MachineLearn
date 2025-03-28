from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

#Um projeto de IA que realiza a leitura e classificação de dígitos escritos a mão

(x_train,y_train),(x_test,y_test)=mnist.load_data()


#Realizando Normalização e unificação das dimensões
x_train = x_train.reshape(-1, 28 * 28)/255.0
x_test = x_test.reshape(-1, 28 * 28)/255.0

#Definindo Tamanho do Batch
batch_size=1024

#Transformando valores de entrada em float32
xph_train=tf.Variable(x_train,dtype=tf.float32)
xph_test=tf.Variable(x_test,dtype=tf.float32)

#Transformando valores da resposta final em inteiros
yph_train=tf.Variable(y_train,dtype=tf.int32)
yph_test=tf.Variable(y_test,dtype=tf.int32)

#Definindo números de neurónios
neuronios_entrada=int(28**2)
neuronios_oculta1=128
neuronios_oculta2=neuronios_oculta1
neuronios_oculta3=neuronios_oculta1
neuronios_saida=10

#Definindo tipo de variavel dos Pesos e da Bias
initializer = tf.keras.initializers.GlorotUniform()

#Definindo Pesos
W = {'oculta1':tf.Variable(initializer([neuronios_entrada,neuronios_oculta1])),
     'oculta2':tf.Variable(initializer([neuronios_oculta1,neuronios_oculta2])),
     'oculta3':tf.Variable(initializer([neuronios_oculta2,neuronios_oculta3])),
     'saida':tf.Variable(initializer([neuronios_oculta3,neuronios_saida]))
     }

#Definindo Bias
bias = {'oculta1':tf.Variable(initializer([neuronios_oculta1])),
     'oculta2':tf.Variable(initializer([neuronios_oculta2])),
     'oculta3':tf.Variable(initializer([neuronios_oculta3])),
     'saida':tf.Variable(initializer([neuronios_saida]))
     }

#Realizando integração com os dados de entradas em batchs
train_dataset = tf.data.Dataset.from_tensor_slices((xph_train,yph_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((xph_test,yph_test))
test_dataset = test_dataset.batch(batch_size)

#Realizando definição da função e perda
def perda(x,y):
    erro = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x,labels=y)
    return tf.reduce_mean(erro)

#Definindo função ativada
otimizador = tf.keras.optimizers.Adam(learning_rate=0.001)

#Começo do Calculo
for epoch in range(10):
     
     epoca_perda=0

     for bacht_x, bacht_y in train_dataset:

          with tf.GradientTape() as tape:     
               
               saida_oculta1=tf.add(tf.matmul(bacht_x,W['oculta1']),bias['oculta1'])
               saida_oculta2=tf.add(tf.matmul(tf.nn.relu(saida_oculta1),W['oculta2']),bias['oculta2'])
               saida_oculta3=tf.add(tf.matmul(tf.nn.relu(saida_oculta2),W['oculta3']),bias['oculta3'])
               saida = tf.add(tf.matmul(saida_oculta3,W['saida']),bias['saida'])

               perde = perda(saida,bacht_y)

          gradiente = tape.gradient(perde,[W['oculta1'],W['oculta2'],W['oculta3'],W['saida'],bias['oculta1'],bias['oculta2'],bias['oculta3'],bias['saida']])

          otimizador.apply_gradients(zip(gradiente,[W['oculta1'],W['oculta2'],W['oculta3'],W['saida'],bias['oculta1'],bias['oculta2'],bias['oculta3'],bias['saida']]))
     
          epoca_perda += perde.numpy()

          
     print('Época: ',epoch,' Erro: ',epoca_perda / len(train_dataset))

#Captura do Peso final e do Bias Final com base no menor erro verificado
W_final,bias_final=[W,bias]

#Refazendo processo de classificação do valor. Para testar outros Números, seria apenas necessário alterar o valor dentro de xph_test[X].
x_test_reshepad = tf.expand_dims(xph_test[1],axis=0)                                              #                                   ^
saida_oculta1=tf.add(tf.matmul(x_test_reshepad,W_final['oculta1']),bias_final['oculta1'])
saida_oculta2=tf.add(tf.matmul(tf.nn.relu(saida_oculta1),W_final['oculta2']),bias_final['oculta2'])
saida_oculta3=tf.add(tf.matmul(tf.nn.relu(saida_oculta2),W_final['oculta3']),bias_final['oculta3'])
saida = tf.add(tf.matmul(saida_oculta3,W_final['saida']),bias_final['saida'])

#Mostrando o resultado da classificação da máquina
print('Classificação da máquina: ',tf.argmax(saida,1).numpy()[0],"\n")

#Mostrando real valor da numeração em imagem. Caso venha a alterar o xph_test[X] acima, necessitará altera o mesmo também na linha 103, para a mesma numeração.
plt.imshow(xph_test[1].numpy().reshape(28,28)*255, cmap="gray")
plt.show()
