from json import load
import pandas as pd
import numpy as np

from tqdm import trange
from mat4py import loadmat
from matplotlib import pyplot as plt
from modules.Activations import sigmoid,ReLU
from modules.ActivationsDerivatives import sigmoid_prime,ReLU_prime
from modules.ImageVectorProcessor import show_images,show_errors,chopImage
from modules.Optimizers import adam
from modules.Tmatrix import Tmatrix,TAvgMatrix

Q1_DATA = loadmat('./data/data21.mat')
Q2_DATA = loadmat('./data/data22.mat')
Q3_DATA = loadmat('./data/data23.mat')

N = 300
EPOCHS = 50 
learning_rate = 0.001

def getArrays(data):
    A1_df = pd.DataFrame(data["A_1"])
    A2_df = pd.DataFrame(data["A_2"])
    B1_df = pd.DataFrame(data["B_1"])
    B2_df = pd.DataFrame(data["B_2"])

    A1 = A1_df.to_numpy()
    A2 = A2_df.to_numpy()
    B1 = B1_df.to_numpy()
    B2 = B2_df.to_numpy()

    return A1, A2, B1, B2

def loadNoisedImages(nImages, N=0, data = Q2_DATA,resolution=(28,28)):
    idealImages = []
    choppedNoisedImage = []

    for i in range(nImages):
        Xi_df = pd.DataFrame(data["X_i"]).iloc[:, i]
        Xn_df = pd.DataFrame(data["X_n"]).iloc[:, i]

        Xi = Xi_df.to_numpy()
        X_n = Xn_df.to_numpy()

        X_i = np.reshape(Xi, (28, 28))
        X_i = np.transpose(X_i)
        if N!=0 : 
            X_n = np.array(chopImage(Xn, N))
        X_n = np.reshape(X_n, resolution)
        X_n = np.transpose(X_n)

        idealImages.append(X_i)
        choppedNoisedImage.append(X_n)
    return idealImages, choppedNoisedImage

def outputCalculator(A1, A2, B1, B2, Zeta=None):
    Z = np.random.normal(loc=0.0, scale=1.0, size=(10, 1)) if Zeta==None else Zeta

    W1 = np.transpose(np.matmul(A1, Z) + B1)
    Z1 = np.maximum(W1, 0)
    W2 = np.matmul(A2, np.transpose(Z1)) + B2
    X = sigmoid(W2)

    return X

def generator(nImages, data):
    images1DList = []
    images2DList = []

    A1, A2, B1, B2 = getArrays(data)

    for _ in range(nImages):
        X = outputCalculator(A1, A2, B1, B2)
        images1DList.append(X)

        X = np.reshape(X, (28, 28))
        X = np.transpose(X)
        images2DList.append(X)

    return images1DList, images2DList

def Q1(imageData,nImageSamples = 100):
    _ , images2D = generator(nImageSamples, imageData)
    show_images(images2D,10,10)

def Q2(image_samples,N,index):
    Xi, Xn = loadNoisedImages(image_samples, N)
    A1, A2, B1, B2 = getArrays(Q1_DATA)
    Z = np.random.normal(loc=0.0, scale=1.0, size=(10, 1))
    T = Tmatrix(N)

    Xn_starting = Xn[index]

    for epoch in range(EPOCHS):

        W1 = np.transpose(np.matmul(A1,Z) + B1)
        Z1 = np.maximum(W1, 0)
        W2 = np.matmul(A2, np.transpose(Z1)) + B2
        X = sigmoid(W2)

        Xn = np.matmul(T, np.reshape(Xn_starting, (784, 1)))

        Jz = np.log((np.linalg.norm(Xn - np.matmul(T, X))) ** 2)+ (np.linalg.norm(Z)) ** 2

        U2 = (- 2 * np.dot(np.transpose(T), (Xn - np.dot(T, X))))/(np.log(10) * (np.linalg.norm(Xn - np.dot(T, X))) ** 2)
        V2 = np.multiply(U2, sigmoid_prime(W2))
        U1 = np.matmul(np.transpose(A2), V2)
        V1 = np.multiply(U1, ReLU_prime(np.transpose(W1)))
        U0 = np.matmul(np.transpose(A1), V1)

        Grad = N*U0 + 2*Z

        Z = adam(epoch, initial_Z=Z, gradient=Grad,learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        # print('Error : {}'.format(Jz))

    W1 = np.transpose(np.matmul(A1,Z) + B1)
    Z1 = np.maximum(W1, 0)
    W2 = np.matmul(A2, np.transpose(Z1)) + B2
    X = sigmoid(W2)

    X = np.reshape(X, (28, 28))

    X = np.transpose(X)

    Xn_starting = np.reshape(Xn_starting, (28, 28))

    Xi_initial = np.reshape(Xi[index], (28, 28))
        
    return [Xi_initial,Xn_starting,X]

        
def Q3():
    Xi, Xn = loadNoisedImages(4,0, Q3_DATA, (7,7))
    A1, A2, B1, B2 = getArrays(Q1_DATA)
    errors = []
    res = []
    for Xi, Xn in zip(Xi, Xn):
        Z = np.random.normal(loc=0.0, scale=1.0, size=(10, 1))
        
        Xn_starting = Xn
        temp_error_list = []
        for epoch in trange(EPOCHS):

            W1 = np.transpose(np.matmul(A1, Z) + B1)
            Z1 = np.maximum(W1, 0)
            W2 = np.matmul(A2, np.transpose(Z1)) + B2
            X = sigmoid(W2)

            T = TAvgMatrix()

            Xn_for_calcs = np.reshape(Xn_starting, (49, 1))

            Jz = np.log((np.linalg.norm(Xn_for_calcs - np.matmul(T,X))) ** 2)
            + (np.linalg.norm(input)) ** 2
            Xn = Xn_for_calcs
            U2 = (- 2 * np.dot(np.transpose(T), (Xn - np.dot(T, X))))/(np.log(10) * (np.linalg.norm(Xn - np.dot(T, X))) ** 2)


            V2 = np.multiply(U2, sigmoid_prime(W2))


            U1 = np.matmul(np.transpose(A2), V2)


            V1 = np.multiply(U1, ReLU_prime(np.transpose(W1)))


            U0 = np.matmul(np.transpose(A1), V1)


            Grad = U0 + 2*input
            Z = adam(epoch, initial_Z=Z, gradient=Grad,learning_rate=learning_rate, beta1=0.9, beta2=0.999)
            
            temp_error_list.append(Jz)

        errors.append(temp_error_list)


        W1 = np.transpose(np.matmul(A1, Z) + B1)
        Z1 = np.maximum(W1, 0)
        W2 = np.matmul(A2, np.transpose(Z1)) + B2
        X = sigmoid(W2)

        X = np.reshape(X, (28, 28))
        X = np.transpose(X)
        
        Xn_starting = np.reshape(Xn_starting, (7, 7))
        Xi_initial = np.reshape(Xi, (28, 28))
        
        
        res.append(Xi_initial)
        res.append(Xn_starting)
        
        res.append(X)
    show_images(res,3,4)
    show_errors(errors)

if __name__ == "__main__":

    Q1(Q1_DATA)
    
    show_images(Q2(4,500,0) + Q2(4,500,1) + Q2(4,500,2) + Q2(4,500,3),4,3)
    show_images(Q2(4,400,0) + Q2(4,400,1) + Q2(4,400,2) + Q2(4,400,3),4,3)
    show_images(Q2(4,350,0) + Q2(4,350,1) + Q2(4,350,2) + Q2(4,350,3),4,3)
    show_images(Q2(4,300,0) + Q2(4,300,1) + Q2(4,300,2) + Q2(4,300,3),4,3)

    Q3()

