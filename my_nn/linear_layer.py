import numpy as np

class LinearLayer:
    
    '''
    A fully connected linear layer with activation function
    '''
    
    def __init__(self, input_size, output_size, activation_fn = 'ReLU'):
        
        '''
        input_size: int; size of input
        output_size: int; size of output
        activation_fn: str; type of activation function for the layer
        '''
        
        self.weights = np.random.randn(input_size, output_size) # shape (input, output)
        self.weights_grads = np.empty_like(self.weights) #shape (input, output)
        self.biases = np.random.randn(output_size) # shape (output, )
        self.biases_grads = np.empty_like(self.biases) # shape (output, )
        
        
        activation_functions = {
            'relu': {
                'function': self.ReLU,
                'gradient': self.ReLU_backward
            },
            'sigmoid': {
                'function': self.sigmoid, 
                'gradient': self.sigmoid_backward
            }
        }   
        
        self.activation_function = activation_functions[activation_fn.lower()]['function']
        self.activation_function_gradient = activation_functions[activation_fn.lower()]['gradient']
        
    def forward(self, x):
        self.x = x
        self.value = np.matmul(x, self.weights) + self.biases
        self.activation_value = self.activation_function(self.value)
        return self.activation_value      
    
    def backward(self, output_grad):
        activation_grad = self.activation_function_gradient(output_grad)
        self.biases_grads = activation_grad.reshape(self.biases_grads.shape)
        self.weights_grads = np.matmul(self.x.T, activation_grad)
        return np.matmul(activation_grad, self.weights.T)
    
    def step(self, learning_rate=1e-2):
        self.weights -= self.weights_grads * learning_rate
        self.biases -= self.biases_grads * learning_rate
    
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def ReLU_backward(self, output_grad):
        return output_grad * (self.activation_value > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_backward(self, output_grad):
        return output_grad * self.activation_value * (1 - self.activation_value) 