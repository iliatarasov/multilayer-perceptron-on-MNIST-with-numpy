import numpy as np
import functools
from collections import defaultdict

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score

from .linear_layer import LinearLayer       

class MLPClassifier:
    
    '''
    MLP multi-class classifier with linear layers and softmax 
    as the activation function for the last layer and cross-entropy
    as the loss function
    '''
    
    def __init__(self, input_size, output_size, n_layers, layer_sizes, loss_fn = 'softmax_cross_entropy', activation_fns='ReLU'):
        
        '''
        input_size: int; size of 1 instance of data (X)
        
        output_size: int; size of the prediction vector
        
        n_layers: int; number of layers in the network
        
        layer_sizes: list[int], len(layer_sizes) = n_layers - 1; dimensions of layers
        
        loss_fn: str; type of loss function, currently only softmax_cross_entropy is available
        
        activation_fns: str or list[str] (len(list) = n_layers - 1); 
        if str, defines the activation function for all but the last layers, 
        if list, defines activation functions for each layer separately
        '''
        
        if isinstance(activation_fns, str):
            activation_fns = [activation_fns] * n_layers
        elif isinstance(activation_fns, list):
            assert len(activation_fns) == n_layers
        
        layer_sizes = [input_size] + layer_sizes + [output_size]
        
        self.layers = [LinearLayer(layer_sizes[i], layer_sizes[i + 1], 
                                   activation_fn=activation_fns[i]) for i in range(n_layers)]
        
        self.layers[-1].activation_function = lambda x: x
        self.layers[-1].activation_function_gradient = lambda x: x
        
        loss_functions = {
            'softmax_cross_entropy': {
                'function': self.softmax_crossentropy_with_logits,
                'gradient': self.grad_softmax_crossentropy_with_logits
            },
        }
        

        self.loss_function = loss_functions[loss_fn]['function']
        self.loss_function_grad = loss_functions[loss_fn]['gradient']
        
        self.metrics = defaultdict(list)

        self.trained = False
        
    def forward(self, x):
        layers = [functools.partial(layer.forward) for layer in self.layers]
        return functools.reduce((lambda x, y: y(x)), layers, x)
        
    def backward(self, output_grad):
        layer_grads = [functools.partial(layer.backward) for layer in self.layers[::-1]]
        return functools.reduce((lambda x, y: y(x)), layer_grads, output_grad)
    
    def softmax_crossentropy_with_logits(self, logits, reference_answers):
        #logits = logits / logits.max()
        softmax = np.exp(logits) / (np.exp(logits).sum() + 1e-6)
        return -np.log(softmax[0, reference_answers] + 1e-6)[0]
    
    def grad_softmax_crossentropy_with_logits(self, logits, reference_answers):
        #logits = logits / logits.max()
        softmax = np.exp(logits) / (np.exp(logits).sum() + 1e-6)
        softmax[0, reference_answers] -= 1
        return softmax    
    
    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)
    
    def fit(self, X_train, y_train, learning_rate=1e-2, n_epochs=100, show_progress=False, *, X_test=None, y_test=None):
        
        train_size = X_train.shape[0]
        
        if X_test is not None:
            test_size = X_test.shape[0]
            
        if show_progress == True:
            show_progress = 1
        
        self.learning_curve = np.zeros(n_epochs)
        self.accuracy_curve = np.zeros(n_epochs)
        
        for epoch in range(1, n_epochs + 1):
            
            y_pred = []
            
            for sample_i in range(train_size):
                
                x = X_train[sample_i].reshape((1, -1))
                
                target = np.array([y_train[sample_i]])
                logits = self.forward(x)
                
                loss = self.loss_function(logits, target)
                loss_grad = self.loss_function_grad(logits, target)
                
                self.backward(loss_grad)

                self.step(learning_rate)

                y_pred.extend(logits.argmax(1))
                
            self.metrics['accuracy_score'].append(accuracy_score(y_train, y_pred))
            self.metrics['balanced_accuracy_score'].append(balanced_accuracy_score(y_train, y_pred))
            self.metrics['recall_score'].append(recall_score(y_train, y_pred, average='micro'))
            self.metrics['precision_score'].append(precision_score(y_train, y_pred, average='micro'))
            #self.metrics['roc_auc_score'].append(roc_auc_score(y_train, y_pred, multi_class='ovr'))
            self.metrics['loss'].append(loss)
            

            if show_progress and epoch % show_progress == 0:

                y_pred_test = []


                    
                if X_test is not None:
                    for sample_i in range(test_size):
                        x = X_test[sample_i].reshape((1, -1))
                        target = np.array([y_test[sample_i]])

                        logits = self.forward(x)
                        y_pred_test.extend(logits.argmax(1))
                    print(f'Epoch: {epoch}, loss: {loss:.3f}, accuracy on train: {accuracy_score(y_train, y_pred):.3f}, accuracy on test: {accuracy_score(y_test, y_pred_test):.3f}')
                else: print(f'Epoch: {epoch}, loss: {loss:.3f}, accuracy on train: {accuracy_score(y_train, y_pred):.3f}')
                
        self.trained = True
        
    def predict(self, X_test):
        assert self.trained, 'The network was never trained'
        y_pred = []
        for sample in X_test:
            X_i = sample.reshape((1, -1))
            logits = self.forward(X_i)
            y_pred.extend(logits.argmax(1))
        return np.array(y_pred)
                