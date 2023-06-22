import matplotlib.pyplot as plt


class PlotMetrics:
    '''Utility class for plotting metrics'''
    def __init__(self, model):
        '''
        Arguments:
            model (MLPClassifier or MLPClassifierTorch): trained model            
            '''
        self.model = model

    def plot(self):
        '''Plot metrics'''
        n_metrics = len(self.model.metrics)
        fig, ax = plt.subplots(n_metrics, 1, figsize=(10, 3.5*n_metrics))
        fig.tight_layout(pad=5)
        for i, metric in enumerate(self.model.metrics):
            n_epochs = len(self.model.metrics[metric])
            ax[i].plot(range(1, n_epochs + 1), self.model.metrics[metric], color='k')
            ax[i].axhline(self.model.metrics[metric][-1], linestyle='--', color='g')
            ax[i].set_title(metric)
            ax[i].set_xlabel('epoch')
