# モジュールの構造：uitils.modules.pyに記述
from efficientnet import EfficientNetB0
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model

class Usermodel_efficientnet(object):
    def __init__(self, cut_size, channel, category_count):
        self.cut_size = cut_size
        self.channel = channel
        self.category_count = category_count
        
    def bottleneck(self, x):
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.category_count, activation='sigmoid')(x)
        return x
    
    def getTrainModel(self):
        input_shape = (self.cut_size["height"], self.cut_size["width"], self.channel)
        print("input_shape : {}".format(input_shape))
        
        # EfficientNetのオプションについては随時調査（やりたいのは１から学習）
        effnet_instance = EfficientNetB0(input_shape = input_shape,  weights='imagenet', include_top=False)
        x = effnet_instance.output
        x = self.bottleneck(x)

        model = Model(inputs=effnet_instance.input, outputs=x)
        return model
