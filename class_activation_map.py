from keras import Model


class ClassActivationMap:
    def __init__(self, model, layer_GAP=-2, layer_dense=-1):
        """
        kerasのmodelから class activation map を計算する

        Parameters
        ---------------
        model : keras.Model
            kerasの学習済みモデル
            ・model.layers[-4] が特徴マップを出力するConv2D層（GlobalAveragePooling2Dの前）
            ・model.layers[-1] が全結合層
        layer_GAP : int or str
            GlobalAveragePooling2D層のインデックス、またはレイヤー名
        layer_dense : int or str
            全結合層のインデックス、またはレイヤー名
        """
        # モデルに含まれるレイヤー名のリストを取得
        layers_name = [l.name for l in model.layers]

        # レイヤー名で指定された場合はインデックスに変換
        if type(layer_GAP) is str:
            layer_GAP = layers_name.index(layer_GAP)
        if type(layer_dense) is str:
            layer_dense = layers_name.index(layer_dense)

        # GlobalAveragePooling2D層の直前を出力とするモデルを作成する
        self.cam_model = Model(inputs=model.input, outputs=model.layers[layer_GAP-1].output)
        # 全結合層の重み、バイアスを作成する
        self.w, self.b = model.layers[layer_dense].get_weights()
    

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