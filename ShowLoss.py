import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

class ShowLoss(tf.keras.callbacks.Callback):
 
    def __init__(self):
        super(ShowLoss, self).__init__()
        self.loss = []

    def on_train_end(self, batch, logs=None):
        fig = plt.figure()
        plt.plot(np.arange(0,len(self.loss)), self.loss, '+', linestyle="-")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.axis(ymin=0)
        plt.show()
        fig.savefig(f'./loss_{datetime.datetime.now().strftime("%d%m%y_%H:%M:%S")}.png')
    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs['loss'])

