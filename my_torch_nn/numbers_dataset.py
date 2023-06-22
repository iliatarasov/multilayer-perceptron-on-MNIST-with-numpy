import pandas as pd
import torch


class NumbersDataset(torch.utils.data.Dataset):
    '''Numbers dataset class'''
    
    def __init__(self, file_name):
        '''
        Arguments:
            file_name (string): Path to csv file        
        '''
        self.numbers = pd.read_csv(file_name)
        self.labels = torch.tensor(self.numbers.iloc[:, 0].values, 
                                   dtype=torch.int64)
        self.data = torch.tensor(self.numbers.iloc[:, 1:].values,
                                 dtype=torch.float32)
        
    def __len__(self):
        return len(self.numbers)
    
    def __getitem__(self, idx):
        '''
        Return order: X (data), y (label)
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist() 
        
        return self.data[idx], self.labels[idx]
    