# %matplotlib inline
import os
import numpy as np
import sys
np.random.seed(2) #for the sake of debuging we set the random seed


class neuralNetwork():
    #constructor of NN, we recieve number of inputs, number of hidden neurons, number of outputs, and our activation and derivate by default
    def __init__(self,
                 input_neurons=2,
                 hidden_neurons1=2,
                 hidden_neurons2=None,
                 nHidden_layers=1,
                 output_neurons=2,
                 activation_function=lambda x: 1 / (1+np.exp(-x)),
                 derivate_function=lambda x: x*(1-x),
                 path = None):
        if path == None:
            if nHidden_layers != 1 and nHidden_layers != 2:
                print "Number of hidden layers possible is 1 or 2!"
                sys.exit(2)

            self.nHidden = nHidden_layers

            if self.nHidden == 1:
                self.wIH = 2*np.random.rand(hidden_neurons1,input_neurons)  -1
                self.wHO = 2*np.random.rand(output_neurons,hidden_neurons1) -1
                self.bIH = 2*np.random.rand(hidden_neurons1,1) -1
                self.bHO = 2*np.random.rand(output_neurons,1)  -1
            if self.nHidden == 2:
                self.wIH = 2*np.random.rand(hidden_neurons1, input_neurons)   -1
                self.wHH = 2*np.random.rand(hidden_neurons2, hidden_neurons1) -1
                self.wHO = 2*np.random.rand(output_neurons, hidden_neurons2)  -1
                self.bIH = 2*np.random.rand(hidden_neurons1, 1) -1
                self.bHH = 2*np.random.rand(hidden_neurons2, 1) -1
                self.bHO = 2*np.random.rand(output_neurons, 1)  -1


        else:
            if not os.path.exists(path):
                print "No data to load!!"
                sys.exit(2)
            self.wIH = np.load(path+"/wIH.npy")
            self.bIH = np.load(path+"/bIH.npy")
            if os.path.isfile(path+"/wHH.npy") and os.path.isfile(path+"/bHH.npy"):
                self.wHH = np.load(path+"/wHH.npy")
                self.bHH = np.load(path+"/bHH.npy")
                self.nHidden = 2
            else:
                self.nHidden = 1
            self.wHO = np.load(path+"/wHO.npy")
            self.bHO = np.load(path+"/bHO.npy")
                    #and we set activation + derivate functions
        self.activation_function = activation_function
        self.derivate_function = derivate_function
    #save de matrix's
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+"/wIH",self.wIH)
        np.save(path+"/bIH",self.bIH)
        if self.nHidden == 2:
            np.save(path+"/wHH", self.wHH)
            np.save(path+"/bHH", self.bHH)
        np.save(path+"/wHO", self.wHO)
        np.save(path+"/bHO", self.bHO)

    #validation, tell neural network to make a prediction
    def validate(self, inputs):
        #feed forward process
        matinputs = np.asmatrix(inputs).T
        h0 = self.apply_function(np.add(np.dot(self.wIH, matinputs), self.bIH), self.activation_function)
        if self.nHidden == 2:
            h1 = self.apply_function(np.add(np.dot(self.wHH, h0), self.bHH), self.activation_function)
            out = self.apply_function(np.add(np.dot(self.wHO, h1), self.bHO), self.activation_function)
        else:
            out = self.apply_function(np.add(np.dot(self.wHO, h0), self.bHO), self.activation_function)
        return np.asarray(out)

    #train the neural network, we tell the inputs, and its correct aligned outputs and the number of times to train
    #and the learning rate as an optional param
    def train(self, inputs, outputs, steps, learning_rate=0.009):
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
            loss = self.step_train(inputs[x], outputs[x], learning_rate, i)
            #if we passed a 1% log the average into the plot!
            totalErr += loss.item(0)**2
            if i%logStep == 0 and i >100:
                lossArray.append(totalErr/i)
                stepArray.append(i)
        #plt.plot(stepArray,lossArray)
        #plt.ylabel('loss evolution')   ####PLOT THE ERROR
        #plt.show() #once we finished we show the plot of the evolution of the error

    def step_train(self, inputs, outputs, learning_rate,i):
        #transpose the input, make it a vertical column
        inputMat = np.asmatrix(inputs).T

        #we make first step of feed forward, dot product weights, input + add the bias and activation function
        h0 = self.apply_function(np.add(np.dot(self.wIH, inputMat), self.bIH), self.activation_function)
        if self.nHidden == 1:
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
        else:
            h1  = self.apply_function(np.add(np.dot(self.wHH, h0), self.bHH), self.activation_function)
            out = self.apply_function(np.add(np.dot(self.wHO, h1), self.bHO), self.activation_function)

            loss = np.asmatrix(outputs).T - out

            hidden_error1 = np.dot(self.wHO.T, loss)
            hidden_error2 = np.dot(self.wHH.T, hidden_error1)

            derivateHO = np.multiply(self.apply_function(out, self.derivate_function), loss)*learning_rate
            derivateHH = np.multiply(self.apply_function(h1, self.derivate_function), hidden_error1)*learning_rate
            derivateIH = np.multiply(self.apply_function(h0, self.derivate_function), hidden_error2)*learning_rate

            deltaWHO = np.dot(derivateHO, h1.T)
            deltaWHH = np.dot(derivateHH, h0.T)
            deltaWIH = np.dot(derivateIH, inputMat.T)

            deltaBHO = derivateHO * 1
            deltaBHH = derivateHH * 1
            deltaBIH = derivateIH * 1

            #update the weights

            self.wHO = np.add(self.wHO, deltaWHO)
            self.wHH = np.add(self.wHH, deltaWHH)
            self.wIH = np.add(self.wIH, deltaWIH)
            #update the bias
            self.bHO = np.add(self.bHO, deltaBHO)
            self.bHH = np.add(self.bHH, deltaBHH)
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
    relu = lambda x: np.maximum(0, x)
    drelu = lambda x: 1 if x > 0 else 0
    tanh = lambda x: (2/(1+np.exp(-2*x))) -1
    dtanh = lambda x: 1 - x**2
    #create a new neural network
    nn = neuralNetwork(
                 input_neurons=4,
                 hidden_neurons1=100,
                 hidden_neurons2=50,
                 nHidden_layers=2,
                 output_neurons=3,
                 activation_function=tanh,
                 derivate_function=dtanh)

   ## nn = neuralNetwork(path="./data", activation_function=tanh, derivate_function=dtanh) #for loading from a data folder

    #generate train set
    inputs = []
    backtracking(0, [0,0,0,0], inputs)
    #generate solutions
    outputs =[]
    for i in inputs:
        x = regular_solver(i)
        outputs.append(x)

    #train neural network
    nn.train(inputs, outputs, 40000)
    #validate simple value
    for i in inputs:
        print i
        print nn.validate(i)
    ##nn.save("./data") ## save the weights of your current nn into ./data folder, you can specify the folder you want, it will be created
if __name__ == "__main__":
    run()
