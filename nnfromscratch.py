%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2) #for the sake of debuging we set the random seed

class neuralNetwork():
    #constructor of NN, we recieve number of inputs, number of hidden neurons, number of outputs, and our activation and derivate by default
    def __init__(self, input_neurons, hidden_neurons, output_neurons, activation_function=lambda x: 1 / (1+np.exp(-x)), derivate_function=lambda x: x*(1-x)):
        #we initialize the weight and bias random
        self.wIH = 2*np.random.rand(hidden_neurons,input_neurons) -1
        self.wHO = 2*np.random.rand(output_neurons,hidden_neurons) -1
        self.bIH = 2*np.random.rand(hidden_neurons,1) -1
        self.bHO = 2*np.random.rand(output_neurons,1) -1
        #and we set activation + derivate functions
        self.activation_function = activation_function
        self.derivate_function = derivate_function

    #validation, tell neural network to make a prediction
    def validate(self, inputs):
        #feed forward process
        matinputs = np.asmatrix(inputs).T
        h0 = self.apply_function(np.add(np.dot(self.wIH, matinputs), self.bIH), self.activation_function)
        out = self.apply_function(np.add(np.dot(self.wHO, h0), self.bHO), self.activation_function)
        return np.asarray(out)
    #train the neural network, we tell the inputs, and its correct aligned outputs and the number of times to train
    #and the learning rate as an optional param
    def train(self, inputs, outputs, steps, learning_rate=0.05):
        #we set minimum steps to 1000
        if(steps < 1000):
            print "Training needs more than 1000 steps!"
            return
        #for generating the graphs, every 1% of the whole training we will save to the graph the avg of error
        totalErr = 0
        logStep = steps//100
        lossArray = []
        stepArray = []
        lowered = False
        for i in range(0, steps):
            # Lower the learning rate at half path!
            if i > steps/2 and not lowered:
                lowered = True
                learning_rate = learning_rate / 2
            #get a random input and make the training using stocastic gradient descent
            x = np.random.randint(0, len(inputs))
            loss = self.step_train(inputs[x], outputs[x], learning_rate=learning_rate)
            #if we passed a 1% log the average into the plot!
            if i%logStep == 0 and i >100:
                totalErr += loss.item(0)**2
                lossArray.append(totalErr/i)
                stepArray.append(i)
        plt.plot(stepArray,lossArray)
        plt.ylabel('loss evolution')
        plt.show() #once we finished we show the plot of the evolution of the error

    def step_train(self, inputs, outputs, learning_rate):
        #transpose the input, make it a vertical column
        inputMat = np.asmatrix(inputs).T

        #we make first step of feed forward, dot product weights, input + add the bias and activation function
        h0 = self.apply_function(np.add(np.dot(self.wIH, inputMat), self.bIH), self.activation_function)
        #second step, take the output of hidden layer and make the same process
        out = self.apply_function(np.add(np.dot(self.wHO, h0), self.bHO), self.activation_function)
        #we calculate the loss
        loss = np.asmatrix(outputs).T - out
        #and the hidden error to ponderate weights error
        hidden_error = np.dot(self.wHO.T, loss)

        #calculate ponderated derivative of outside
        #used for calculating 4 delta's
        derivateHO = np.multiply(self.apply_function(out, self.derivate_function), loss)*learning_rate
        derivateIH = np.multiply(self.apply_function(h0, self.derivate_function), hidden_error)*learning_rate

        deltaWHO = np.dot(derivateHO , h0.T)  #chain rule, multiply by partial derivative of  WHO*H + b
        deltaWIH = np.dot(derivateIH , inputMat.T) #chain rule, multiply by partial  derivative of WIH*Inputs + b

        deltaBHO = derivateHO * 1  #chain rule, the derivative of WHO*h + b respect B is 1  #useless
        deltaBIH = derivateIH * 1  #chain rule, the derivative of WIH*h + b respect B is 1  #usseless

        #update the weights
        self.wHO = np.add(self.wHO, deltaWHO)
        self.wIH = np.add(self.wIH, deltaWIH)

        #update the bias
        self.bHO = np.add(self.bHO, deltaBHO)
        self.bIH = np.add(self.bIH, deltaBIH)

        return loss

    def apply_function(self, x, function):

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = function(x[i,j])

        return x


# We will use the neural network class for adding to numbers
# Our input will be [x2, x1, y2, y1] and the ouput will be x+y
#where xn represents a binary bit of x
#############for generating dataset##########################
#generates every combination of binary numbers of 4 bits
def backtracking(i, marcas, solucions):
    if i == 4:
        solucions.append(marcas)
    else:
        backtracking(i+1, marcas[:], solucions)
        marcas[i] = 1
        backtracking(i+1, marcas[:], solucions)
        marcas[i] = 0
#makes the sum of x + y and returns the result in the list
def regular_solver(llista):
    num1 = llista[0]*2+llista[1]*1
    num2 = llista[2]*2+llista[3]*1
    suma = num1 + num2
    result =[0,0,0]
    for i in reversed(range(0, 3)):
        if suma >= 2**i:
            result[2-i] = 1
            suma = suma - 2**i
    return result
########################################################

def run():
    # our activation function
    tanh = lambda x: (2/(1+np.exp(-2*x))) -1
    dtanh = lambda x: 1 - x**2
    #our neural network, 4 input bits, 45 hidden neurons, 3 output bits
    nn = neuralNetwork(4, 45, 3, tanh, dtanh)

    #generate input
    inputs = []
    backtracking(0, [0,0,0,0], inputs)
    #generate solutions
    outputs =[]
    for i in inputs:
        x = regular_solver(i)
        outputs.append(x)
    #train neural network
    nn.train(inputs, outputs, 20000)
    #validate simple value
    print nn.validate([1,0,0,1])
if __name__ == "__main__":
    run()
