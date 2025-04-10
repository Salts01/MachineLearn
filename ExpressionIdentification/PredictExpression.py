import tensorflow as tf 
from tkinter import Tk
from tkinter.filedialog import askopenfile
from PIL import Image
import os

base_path = os.path.dirname(os.path.abspath(__file__))

inicializador=tf.keras.initializers.GlorotUniform() #Definição da tipagem dos pesos

Wconv1=tf.Variable(inicializador([3,3,3,64],dtype=tf.float32)) 
bias1=tf.Variable(inicializador([64],dtype=tf.float32))

Wconv2=tf.Variable(inicializador([3,3,64,128],dtype=tf.float32))
bias2=tf.Variable(inicializador([128],dtype=tf.float32))

Wconv3=tf.Variable(inicializador([3,3,128,128],dtype=tf.float32))
bias3=tf.Variable(inicializador([128],dtype=tf.float32))

Wfinal=tf.Variable(inicializador([16*16*128,4],dtype=tf.float32))
biasFinal=tf.Variable(inicializador([4],dtype=tf.float32))

print('Realizando carregamento de pesos atualizados')

Pesos_atualizados=tf.train.Checkpoint(Wconv1=Wconv1, bias1=bias1, Wconv2=Wconv2, bias2=bias2, Wconv3=Wconv3, bias3=bias3, Wfinal=Wfinal, biasFinal=biasFinal)
Pesos_atualizados.restore(os.path.join(base_path, "Pesos_atualizados/Pesos"))

print("Realizando carregamento com sucesso!")


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


def SelectImage(): # Caixa de diálogo para selecionar a foto
    root=Tk()
    root.withdraw()
    caminho=askopenfile(title='Selecione o A imagem',filetypes=[("Imagens", "*.jpg *.png *.jpeg")],mode='rb')
    return caminho

def Previsao(imagem):
    
    imagem=Image.open(imagem).convert('RGB') # Realizando adição da coluna RGB
    imagem=imagem.resize((128,128)) # Realizando conversão para 128x128
    img_array=tf.Variable(imagem,dtype=tf.float32)/255.0 #Realizando conversão para tipo float 32 e normalização
    entering=tf.expand_dims(img_array,axis=0) #Adcionando coluna de batch


    logits=conv_function(entering)
    predict=tf.argmax(logits,axis=1).numpy()[0]

    classes =  ['feliz',"medo",'raiva','triste']

    print('Classificação: ',classes[predict])




image=SelectImage()
Previsao(image)


