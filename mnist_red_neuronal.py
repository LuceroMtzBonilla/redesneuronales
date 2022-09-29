# -*- coding: utf-8 -*-
import mnist_loader
import network
import pickle

#Dividimos los datos 
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

#le ponemos el formato de lista
training_data = list(training_data)
test_data = list(test_data)

#definimos la red neuronal con una capa de entrada de 784 neuronas, una capa intermedia de 
# 30 neuronas y una capa de salida con 10 neuronas (porque hay 10 digitos a reconocer)
net = network.Network([784,30,10])



#ENTRENAMIENTO 
#usamos el algoritmo Stochastic Gradient Descent para entrenar el modelo, se 
#usaron los datos de entrenamiento con 30 epocas, 15 mini batch y  un learning rate de 3.0
net.SGD( training_data, 30, 15, 3.0, test_data=test_data)


#archivo = open("red_prueba_RMS.pkl",'wb')

#pickle.dump(net,archivo)
#archivo.close()
#exit()
#leer el archivo
#archivo_lectura = open("red_prueba_RMS_1.pkl",'rb')
#net = pickle.load(archivo_lectura)
#archivo_lectura.close()
#net.SGD( training_data, 10, 50, 0.5, test_data=test_data)
#archivo = open("red_prueba_RMS_1.pkl",'wb')
#pickle.dump(net,archivo)

#archivo.close()
#exit()


