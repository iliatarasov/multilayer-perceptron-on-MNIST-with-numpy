import numpy as np


class LinearLayer:
    '''
    A linear layer with ReLU activation function
    '''
    
    def __init__(self, input_size, output_size):
        
        '''
        input_size: int; size of input
        output_size: int; size of output
        activation_fn: str; type of activation function for the layer
        '''
        
        self.weights = np.random.randn(input_size, output_size)/10. # shape (input, output)
        self.weights_grads = np.empty_like(self.weights) #shape (input, output)
        self.biases = np.random.randn(output_size)/10. # shape (output, )
        self.biases_grads = np.empty_like(self.biases) # shape (output, )
        
        self.activation_function = self.ReLU
        self.activation_function_gradient = self.ReLU_backward
        
    def forward(self, x):
        '''Forward pass'''
        self.x = x
        self.value = np.matmul(x, self.weights) + self.biases
        self.activation_value = self.activation_function(self.value)
        return self.activation_value      
    
    def backward(self, output_grad):
        '''Backpropagation'''
        activation_grad = self.activation_function_gradient(output_grad)
        self.biases_grads = activation_grad.reshape(self.biases_grads.shape)
        self.weights_grads = np.matmul(self.x.T, activation_grad)
        return np.matmul(activation_grad, self.weights.T)
    
    def step(self, learning_rate=1e-3):
        '''Learning rate application'''
        self.weights -= self.weights_grads * learning_rate
        self.biases -= self.biases_grads * learning_rate
    
    def ReLU(self, x):
        '''Relu activation function'''
        return np.maximum(0, x)
    
    def ReLU_backward(self, output_grad):
        '''Relu activation function backpropagation'''
        return output_grad * (self.activation_value > 0).astype(float)