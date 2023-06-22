from torch import nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    '''Architecture class for mlp_classifier'''
    
    def __init__(self, input_size, output_size, n_layers, layer_dims):
        super(NeuralNet, self).__init__()
        '''
        Arguments:
            input_size: (int): size of 1 instance of data (X)
            
            output_size: (int): size of the prediction vector
            
            n_layers: (int): number of layers in the network
            
            layer_dims: list[int]: dimensions of hidden layers,
            (len(layer_dims) = n_layers - 1)
        '''
        layer_dims = [input_size] + layer_dims + [output_size]

        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(n_layers)
        ])

    def forward(self, x):
        '''Forward pass'''
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
