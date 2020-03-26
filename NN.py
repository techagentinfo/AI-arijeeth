import numpy as np

class SNN():
    
    def __init__(self):
        np.random.seed(1)
        self.sy_value = 2 * np.random.random((3, 1)) - 1 #value from -1 to 1 with mean 0

    def sigm_funct(self, x): #takes values and adds it and defines in range of 1 or 0
        return 1 / (1 + np.exp(-x))

    def sigm_val(self, x): #to calculate correct synaptic values
        return x * (1 - x)

    def model(self, in_value, ou_value, model_inc): #training function with values and errors produced
        for inc in range(model_inc):
            output = self.load(in_value) #modelling values passed through nn
            error = ou_value - output #rate of error
            changes = np.dot(in_value.T, error * self.sigm_val(output)) #error*input values*gradient of SF
            self.sy_value += changes #adjusting synaptic values 

    def load(self, inputs): #output of nn
        inputs = inputs.astype(float)
        output = self.sigm_funct(np.dot(inputs, self.sy_value))
        return output


if __name__ == "__main__":

    nn = SNN() # starting nn

    print("synaptic values = ")
    print(nn.sy_value)
    #training values and output
    in_value = np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,1],[1,0,1],[0,1,1]])
    ou_value = np.array([[0,0,1,1,1,0]]).T
    nn.model(in_value, ou_value, 10000)#Model the neural n/w

    print("Synaptic values after modelling : ")
    print(nn.sy_value)

    #inputs
    M = str(input("1st input 1/0: "))
    N = str(input("2nd input 1/0: "))
    O = str(input("3rd input 1/0: "))
    
    print("Input Values = ", M, N, O)
    print("Output Value = ")
    print(nn.load(np.array([M, N, O])))
