import numpy as np
import sys

#e^x/sum of e^x
def softMax(z2):
    zexp = np.exp(z2 - np.max(z2))
    sum = np.sum(zexp)
    return zexp / sum

#derivative Relu
def derivativeRelu(x):
    if (x.all() > 0):
        return 1
    return 0

#normalize x
def normalize(x):
   return x/np.max(x)


#Initialize all the parameters
def initialize():
    #load train_x and train_y
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2])
    train_x = train_x.astype(np.float)
    train_x =normalize(train_x)
    train_y = np.array(train_y)
    train_y = train_y.astype(np.float)
    #shuffle the data
    shuffleData = list(zip(train_x, train_y))
    np.random.shuffle(shuffleData)
    train_x, train_y = zip(*shuffleData)
    #initialize w1,w2,b1,b2 with random values
    w1 = np.random.randn(len(train_x[0]), 256)
    b1 = np.random.randn(1, 256)
    w2 = np.random.randn(256, 10)
    b2 = np.random.randn(1, 1)
    ret = {'W1': w1, 'W2': w2, 'b1': b1, 'b2': b2, 'Train_x': train_x, 'Train_y': train_y}
    return ret


#Train the algoritem
def train():
    parameters = initialize()
    w1, w2, b1, b2, train_x, train_y = [parameters[key] for key in ('W1', 'W2', 'b1', 'b2', 'Train_x', 'Train_y')]
    eta = 0.005
    #Run 10 epoch
    for i in range(10):
        for x, y in zip(train_x, train_y):
            #calculate the hidden layer
            z1 = np.dot(x, w1) + b1
            h1 = ReLu(z1)
            h1=normalize(h1)
            z2 = np.dot(h1, w2) + b2
            h2 = softMax(z2)
            y = int(y)
            #calculate the loss
            loss = -1 * (np.log(h2[0][y]))
            ret = {'x': x, 'y': y, 'z1': z1, 'z2': z2, 'h1': h1, 'h2': h2, 'loss': loss, 'W1': w1, 'W2': w2, 'b1': b1,
                   'b2': b2}
            #Calculate backprop and update the weightd
            result = backProp(ret)
            rW1, rW2, rb1, rb2 = [result[key] for key in ('W1', 'W2', 'b1', 'b2')]
            w1 = np.subtract(w1,eta * rW1)
            w2 = np.subtract(w2 , eta * rW2)
            b1 = np.subtract(b1 , eta * rb1)
            b2 = np.subtract(b2 , eta * rb2)
    result = {'W1': w1, 'W2': w2, 'b1': b1, 'b2': b2}
    return result


def backProp(params):
    x, y, z1, z2, h1, h2, loss = [params[key] for key in ('x', 'y', 'z1', 'z2', 'h1', 'h2', 'loss')]
    newY = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    newY = newY.astype(np.int)
    newY[y] = 1
    #y_hat-y
    dz2 = np.subtract(h2, newY)
    newYT = np.transpose(np.matrix(h1))
    newYT = np.array(newYT)
    #(y_hat-y)*h1
    dw2 = np.multiply(newYT, dz2)
    db2 = dz2
    w2T = np.transpose(np.matrix(params['W2']))
    w2T = np.array(w2T)
    #(y_hat-y)*h1*relu'(z1|)
    dz1 = (np.dot(db2, w2T)) * derivativeRelu(z1)
    xT = np.transpose(np.matrix(x))
    xT = np.array(xT)
    #    #(y_hat-y)*h1*relu'(z1|)*x
    dw1 = np.multiply(xT, dz1)
    db1 = dz1
    return {'b1': db1, 'W1': dw1, 'b2': db2, 'W2': dw2}


def ReLu(x):
    return np.maximum(0, x)


def test(finalParams):
    #load test_x
    test_file = np.loadtxt(sys.argv[3])
    test_file = np.array(test_file)
    test_file=normalize(test_file)
    w1, w2, b1, b2 = [finalParams[key] for key in ('W1', 'W2', 'b1', 'b2')]
    file = open("test_y", "w")
    for x in zip(test_file):
        # calculate the hidden layer
        z1 = np.dot(x, w1) + b1
        h1 = ReLu(z1)
        h1 = normalize(h1)
        z2 = np.dot(h1, w2) + b2
        h2 = softMax(z2)
        #write the prediction to test_y
        y_hat = np.argmax(h2)
        file.write(str(y_hat) + "\n")


def main():
    finalParams = train()
    test(finalParams)


main()
