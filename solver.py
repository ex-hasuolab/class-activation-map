from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, History, TensorBoard

class Solver(object):
    def __init__(self, dataloader_instance, model_instance, batch_size, n_epochs):
        self.dataloader_instance = dataloader_instance
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
        print("x_train shape {}".format(self.dataloader_instance.x_train.shape))
        print("y_train shape {}".format(self.dataloader_instance.y_train.shape))
        
        """
        # 後で監視が必要ならHistory設定
        # history = Histories(self.epochs, self.initial_lr, self.drop, self.epochs_drop, self.accuracy, self.loss, self.learning_rate)

        # 学習率をいじる必要があればLearningRateScheduler(lrs)でコントロール
        # lr_schedule = LearningRateScheduler(lrs)
        """

        # Modelcheckpoint(filepath, monitor, verbose, save_best_only : 最良モデルをもつ, save_wights_only : Falseなら全体が保存される，period : チェックポイント間のエポック)
        model_checkpoint = ModelCheckpoint("final_weights_step.hdf5", monitor = 'val_loss', verbose = 1,
                                  save_best_only = True, save_weights_only = True, period = 1)
        
        # Tensorboardによる可視化
        log_dir = "./log/"
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

        callbacks = []
        callbacks.append(model_checkpoint)
        callbacks.append(tensorboard)

        model.fit(self.dataloader_instance.x_train, self.dataloader_instance.y_train, batch_size=self.batch_size, \
                  epochs=self.n_epochs, verbose=1, 
                  validation_data=(self.dataloader_instance.x_valid, self.dataloader_instance.y_valid),
                  callbacks = callbacks)