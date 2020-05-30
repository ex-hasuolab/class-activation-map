from keras import Model


class ClassActivationMap:
    def __init__(self, model):
        """
        kerasのmodelから class activation map を計算する

        Parameters
        ---------------
        model : keras.Model
            kerasの学習済みモデル
            ・model.layers[-4] が特徴マップを出力するConv2D層（GlobalAveragePooling2Dの前）
            ・model.layers[-1] が全結合層
        """
        # 中間層を出力とするモデルを作成する
        self.cam_model = Model(inputs=model.input, outputs=model.layers[-4].output)
        # 全結合層の重み、バイアスを作成する
        self.w, self.b = model.layers[-1].get_weights()
    

    def get_class_activation_map(self, images):
        """
        Class Activation Map の取得

        Parameters
        --------------
        images : numpy.array, shape=(n_batches, n_heights, n_width, n_rgb)
            kerasモデルへの入力画像
            shapeは (バッチサイズ, 高さ, 幅, 色)
        
        Returns
        -------------
        cam_images : numpy.array, shape=(n_batches, n_heights, n_width, n_class)
            Class Activation Map
            shapeは (バッチサイズ, 高さ, 幅, クラス数)
        """
        # 入力画像から特徴マップを計算
        feature_map = self.cam_model.predict(images)
        # class activation mapの計算
        cam_images = feature_map.dot(self.w) + self.b
        return cam_images