from collections import defaultdict

import torch
from torch import nn
import numpy as np
import torchmetrics
import pandas as pd

from .neural_net import NeuralNet


class MLPClassifierTorch:
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
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                              else 'cpu')
        self.model = NeuralNet(input_size, output_size, 
                              n_layers, layer_dims).to(self.device)
        self.output_size = output_size

        
    def fit(self, data_loader, learning_rate=1e-3,
            n_epochs=20, show_progress=True):
        '''Training routine'''
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate)

        self.metrics = defaultdict(list)
        balanced_accuracy_score = torchmetrics.Accuracy(task='multiclass', 
                                                        num_classes=self.output_size).to(self.device)
        precision_score = torchmetrics.Precision(task='multiclass', 
                                                 num_classes=self.output_size).to(self.device)
        recall_score = torchmetrics.Recall(task='multiclass', 
                                           num_classes=self.output_size).to(self.device)

        for epoch in range(1, n_epochs+1):
            self.model.train()
            running_loss = []

            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                running_loss.append(loss.cpu().detach().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            predictions = []
            ygt = []
            
            with torch.no_grad():
                for images, labels in data_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    _, y_pred = torch.max(outputs, 1)
                    predictions.append(y_pred)
                    ygt.append(labels)
                    
                ygt = torch.cat(ygt, dim=0)
                predictions = torch.cat(predictions, dim=0)
                    
                self.metrics['balanced accuracy'].append(balanced_accuracy_score(predictions, ygt).cpu())
                self.metrics['recall'].append(recall_score(predictions, ygt).cpu())
                self.metrics['precision'].append(precision_score(predictions, ygt).cpu())
                self.metrics['loss'].append(np.mean(running_loss))

            if show_progress and (epoch == 1 or epoch % show_progress == 0):
                print(f'Epoch: {epoch}/{n_epochs}\tloss: {np.mean(running_loss):.3f}\t',\
                      f'balanced accuracy on train: {self.metrics["balanced accuracy"][-1]:.3f}')

        
    def predict(self, X_test):
        '''Forward pass that returns predictions from test data'''
        self.model.eval()
        if type(X_test) is pd.core.frame.DataFrame:
            X_test = torch.tensor(X_test.values, dtype=torch.float32)
        X_test = X_test.to(self.device)
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu()
