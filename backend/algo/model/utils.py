import keras


class TrainHistory(keras.callbacks.Callback):
    def __init__(self, model_name=None):
        super(TrainHistory, self).__init__()
        self.model_name = model_name
        self.epoch = 0
        self.info = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        message = f"begin epoch: {self.epoch}"
        print(message)

    def on_epoch_end(self, epoch, logs={}):
        message = f'end epoch: {epoch} loss:{logs["loss"]} val_loss:{logs["val_loss"]} acc:{logs["crf_viterbi_accuracy"]} val_acc:{logs["val_crf_viterbi_accuracy"]}'
        print(message)
        dict = {
            'model_name':self.model_name,
            'epoch': self.epoch+1,
            'loss': logs["loss"],
            'acc': logs['crf_viterbi_accuracy'],
            'val_loss': logs["val_loss"],
            'val_acc': logs['val_crf_viterbi_accuracy']
        }
        self.info.append(dict)

    def on_batch_end(self, batch, logs={}):
        message = f'{self.model_name} epoch: {self.epoch} batch:{batch} loss:{logs["loss"]}  acc:{logs["crf_viterbi_accuracy"]}'
        print(message)

