#A02231889
#Akhil Gudavalli
#TOC Assignment 3

#Change the input which is the parameter passed to the fit method in the last line of the program 

import numpy as np
import pickle


class ANN:
    
    def __init__(self,a_input,a_hidden,a_output):
        self.a_input = a_input
        self.a_hidden = a_hidden
        self.a_output = a_output
        
    def build(self):
        self.b_input = len(self.a_input[0])
        self.b_output = len(self.a_output[0])
        np.random.seed(123456)
        self.hidden_weight=np.random.random((self.b_input,self.a_hidden))
        self.hidden_bias=np.random.random((1,self.a_hidden))
        self.output_weight=np.random.random((self.a_hidden,self.b_output))
        self.output_bias=np.random.random((1,self.b_output))
        
    def train(self,iterations):
        self.mu = 0.15
        for i in range(iterations):
            self.forward_propagate(self.a_input)                      
            self.back_propagate()
             
    def fit(self,input_neuron):
        self.input_neuron = input_neuron
        return self.forward_propagate(self.input_neuron)
            
    def forward_propagate(self,input_val):
        self.hidden_values=self.sigmoid((np.dot(input_val,self.hidden_weight)+self.hidden_bias))
        self.output_values=self.sigmoid((np.dot(self.hidden_values,self.output_weight)+self.output_bias))
        return self.output_values
        
    def back_propagate(self):
        self.output_slope = self.derivatives_sigmoid(self.output_values)
        self.hidden_slope = self.derivatives_sigmoid(self.hidden_values)
        
        self.output_error = self.a_output-self.output_values
        self.output_delta = self.output_error * self.output_slope
        
        self.hidden_error = self.output_delta.dot(self.output_weight.T)
        self.hidden_delta = self.hidden_error * self.hidden_slope
        
        self.output_weight += self.hidden_values.T.dot(self.output_delta)*self.mu
        self.output_bias += np.sum(self.output_delta) *self.mu
        
        self.hidden_weight += self.a_input.T.dot(self.hidden_delta)*self.mu
        self.hidden_bias += np.sum(self.hidden_delta) *self.mu
        
    def sigmoid (self,x):
        return 1/(1 + np.exp(-x))

    def derivatives_sigmoid(self,x):
        return x * (1 - x)
    
    def save(self,filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
def restore(filename):
    file = open(filename,'rb')
    object_file = pickle.load(file)
    return object_file
        
                    
                        
a_input = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])

a_output = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
    
a_hidden = 3

A = ANN(a_input,a_hidden,a_output)
A.build()
A.train(100000)
print("\nUsing the Original Object....")
print(A.fit(np.array([0,0,0,1,0,0,0,0])))
A.save('output.obj')
B = restore('output.obj')
print("\nUsing the Restored Object....")
print (B.fit(np.array([0,0,0,1,0,0,0,0])))
