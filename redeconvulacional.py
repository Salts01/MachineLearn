import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Adicionando mais uma coluna, a escala de cinza. A adição nesta forma sempre vai ser na escala de cinza devido ao valor sempre vir 1
x_train=x_train[...,tf.newaxis]
x_test=x_test[...,tf.newaxis]

xph_test=tf.Variable(x_test,dtype=tf.float32)
xph_train=tf.Variable(x_train,dtype=tf.float32)

yph_train=tf.Variable(y_train,dtype=tf.int32)
yph_test=tf.Variable(y_test,dtype=tf.int32)

model=keras.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(10,activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.fit(xph_train,yph_train,epochs=5,batch_size=128,validation_data=(xph_test,yph_test))

teste_loss,test_acc=model.evaluate(xph_test,yph_test, verbose=1)

print('precisão: ',test_acc)