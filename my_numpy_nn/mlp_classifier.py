import numpy as np
import functools
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score

from .linear_layer import LinearLayer      


class MLPClassifier:
    
    '''
    MLP multiclass classifier with an arbitrary number
     of fully connected linear layers and softmax cross-entropy loss function
    '''
    
    def __init__(self, input_size, output_size, n_layers, 
                 layer_dims):
        
        '''
        Arguments:
            input_size: (int): size of 1 instance of data (X)
            
            output_size: (int): size of the prediction vector
            
            n_layers: (int): number of layers in the network
            
            layer_dims: list[int]: dimensions of hidden layers,
            (len(layer_dims) = n_layers - 1)
        '''
    
        
        layer_dims = [input_size] + layer_dims + [output_size]
        
        self.layers = [LinearLayer(layer_dims[i], layer_dims[i + 1]) 
                                   for i in range(n_layers)]
        
        self.layers[-1].activation_function = lambda x: x
        self.layers[-1].activation_function_gradient = lambda x: x        

        self.loss_function = self.softmax_crossentropy
        self.loss_function_grad = self.grad_softmax_crossentropy
        
        self.metrics = defaultdict(list)

        self.trained = False
        
    def forward(self, x):
        '''Forward pass'''
        layers = [functools.partial(layer.forward) for layer in self.layers]
        return functools.reduce((lambda x, y: y(x)), layers, x)
        
    def backward(self, output_grad):
        '''Backpropagation'''
        layer_grads = [functools.partial(layer.backward) 
                       for layer in self.layers[::-1]]
        return functools.reduce((lambda x, y: y(x)), layer_grads, output_grad)
    
    def softmax_crossentropy(self, y, ygt):
        '''Softmax activation and cross-entropy loss'''
        softmax = np.exp(y) / (np.exp(y).sum() + 1e-6)
        return -np.log(softmax[0, ygt] + 1e-6)[0]
    
    def grad_softmax_crossentropy(self, y, ygt):
        '''Softmax activation and cross-entropy loss backpropagation'''
        softmax = np.exp(y) / (np.exp(y).sum() + 1e-6)
        softmax[0, ygt] -= 1
        return softmax    
    
    def step(self, learning_rate):
        '''Application of learning rate'''
        for layer in self.layers:
            layer.step(learning_rate)
    
    def fit(self, X_train, y_train, learning_rate=1e-3, 
            n_epochs=20, show_progress=True):
        '''Training routine'''
        train_size = X_train.shape[0]
            
        if show_progress == True:
            show_progress = 1
        
        for epoch in range(1, n_epochs + 1):
            y_pred = []
            running_loss = []
            
            for sample in range(train_size):
                x = X_train[sample].reshape((1, -1))
                
                ygt_i = np.array([y_train[sample]])
                y_i = self.forward(x)
                
                loss = self.loss_function(y_i, ygt_i)
                loss_grad = self.loss_function_grad(y_i, ygt_i)
                running_loss.append(loss)
                
                self.backward(loss_grad)
                self.step(learning_rate)
                y_pred.extend(y_i.argmax(1))
                
            self.metrics['accuracy'].append(accuracy_score(y_train, y_pred))
            self.metrics['balanced accuracy'].append(balanced_accuracy_score(y_train, y_pred))
            self.metrics['recall'].append(recall_score(y_train, y_pred, average='micro'))
            self.metrics['precision'].append(precision_score(y_train, y_pred, average='micro'))
            self.metrics['loss'].append(np.mean(running_loss))
            
            if show_progress and epoch % show_progress == 0:
                print(f'Epoch: {epoch}/{n_epochs}\tloss: {np.mean(running_loss):.3f}\t',\
                      f'balanced accuracy on train: {balanced_accuracy_score(y_train, y_pred):.3f}')
                
        self.trained = True
        
    def predict(self, X_test):
        '''Forward pass that returns predictions from test data'''
        assert self.trained, 'The network was never trained'
        y_pred = []
        for sample in X_test:
            X_i = sample.reshape((1, -1))
            y_i = self.forward(X_i)
            y_pred.extend(y_i.argmax(1))
        return np.array(y_pred)
                