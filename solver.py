from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, History

class Solver(object):
    def __init__(self, dataloder_instance, model_instance, batch_size, n_epochs):
        self.dataloder_instance = dataloder_instance
        self.model_instance = model_instance
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
    def get_model(self):
        self.model = self.model_instance.getTrainModel()
        # mnist : 'categorical_crossentropy'
        # solar : "binary_crossentropy"
        # optimizer = daget_adam_for_fine_tuning(lr=1e-3, decay=1e-5, multiplier=0.01, model=model)
        self.model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['categorical_accuracy'])
        return self.model
        
    def train(self, model):
        # callbackを定義する場合はここに記述
        print("x_train shape {}".format(self.dataloder_instance.x_train.shape))
        print("y_train shape {}".format(self.dataloder_instance.y_train.shape))

        #history = Histories(self.epochs, self.initial_lr, self.drop, self.epochs_drop, self.accuracy, self.loss, self.learning_rate)
        #lr_schedule = LearningRateScheduler(lrs)
        #model_checkpoint = ModelCheckpoint("final_weights_step1.hdf5", monitor = 'val_loss', verbose = 1,
         #                                   save_best_only = True, save_weights_only = True, period = 1)

        model.fit(self.dataloder_instance.x_train, self.dataloder_instance.y_train, batch_size=self.batch_size, \
                  epochs=self.n_epochs, verbose=1, validation_data=(self.dataloder_instance.x_valid, self.dataloder_instance.y_valid))