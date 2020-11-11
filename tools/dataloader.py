# Dataloader ： utils/loder.pyに記述するべき内容
import numpy as np
import cv2
from math import ceil
from scipy import ndimage
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
import keras
#from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.datasets.mnist import load_data
from keras.preprocessing.image import ImageDataGenerator
from .object_detection import xml_to_labels
import os

class Dataloader(object):
    """
    Attributes
    ------------
    get_mnist_data関数：
        x_train, x_valid, x_test : 訓練用，評価用，テスト用入力画像
        y_train, y_valid, y_test : 訓練用，評価用，テスト用出力ラベル
    
    get_user_data関数：
        train_generator(, validation_generator, test_generator): 訓練用(, 評価用, テスト用)ジェネレータ
    """

    def __init__(self):
        pass
    
    def get_user_data(self, train_dir, validation_dir=None, test_dir=None,
                      batch_size=10,
                      test_batch_size=1,
                      resize_shape=(255, 255),
                      rescale=None,
                      shear_range=0.0,
                      zoom_range=0.0,
                      horizontal_flip=True,
                      shuffle=True):
        '''
        ディレクトリ名を指定して画像データのジェネレータを生成する

        Paramteres
        --------------
        train_dir : str
            訓練データのディレクトリへのパス
        validation_dir : str
            検証データのディレクトリへのパス
        test_dir : str
            テストデータのディレクトリへのパス
        batch_size : int
            訓練データ（と検証データ）のバッチサイズ
        test_batch_size : int
            テストデータのバッチサイズ（デフォルトは1）
        resize_shape : tuple
            画像データをこのサイズにリサイズする
        rescale : float
            画素値のリスケーリング係数。Noneか0ならば適用しない
        shear_range : float
            シアー強度（半時計周り） TODO BBOX情報と関連させるのは難しい？
        zoom_range : float or [float, float]
            ランダムにズームする範囲。
            リストを与えた場合は [lower, upper]。floatを与えた場合は [lower, upper]=[1-zoom_range, 1+zoom_range] 
            TODO BBOX情報と関連させるのは難しい？
        horizontal_flip : bool
            水平方向に入力をランダムに反転するかどうか
        shuffle : bool
            データをシャッフルするかどうか
        
        Notes
        --------------
        train_dirに指定するディレクトリは以下の構造にする。train直下のディレクトリ名がクラス名として認識される
        (validation_dir, test_dirも同様)
        train/
            ┣ class1
            ┃   ┣ class1_1.jpg
            ┃   ┗ class1_2.jpg
            ┣ class2
            ┃   ┣ class2_1.jpg
            ...
        '''
        # 訓練データ
        train_datagen = ImageDataGenerator(
                            rescale=rescale,
                            shear_range=shear_range,  # これ以降は多分水増し関係の設定
                            zoom_range=zoom_range,
                            horizontal_flip=horizontal_flip  # 画像を取得するときにランダムに反転する
                        )

        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=resize_shape,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=shuffle
        )
        
        # 検証データ
        if validation_dir != None:
            validation_datagen = ImageDataGenerator(rescale=rescale)

            self.validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=resize_shape,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=shuffle
            ) 
        
        # テストデータ
        if test_dir != None:
            test_datagen = ImageDataGenerator(rescale=rescale)

            self.test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=resize_shape,
                batch_size=test_batch_size,
                class_mode='categorical',
                shuffle=shuffle
            )
    
    def get_mnist_data(self, resize_mode = False, resize_shape = None, cvtColor_mode = False):
        # load MNIST data
        (x_train, y_train), (x_test, y_test) = load_data()
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.175)
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
        self.x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1).astype('float32')/255
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

        # resize
        if resize_mode == True:
            self.x_train = self.resize(self.x_train, shape = resize_shape)
            self.x_valid = self.resize(self.x_valid, shape = resize_shape)
            self.x_test = self.resize(self.x_test, shape = resize_shape)
            
        # channel がRGBの時（リサイズはどう実装するかは分からない．まとめて出来る？）
        if cvtColor_mode == True:
            self.x_train = self.gray2color(self.x_train)
            self.x_valid = self.gray2color(self.x_valid)
            self.x_test = self.gray2color(self.x_test)
            
        # convert one-hot vector
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_valid = keras.utils.to_categorical(y_valid, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)
    
        # データ数を絞る
        self.x_train = self.x_train[0:500]
        self.x_valid = self.x_valid[0:500]
        self.x_test = self.x_test[0:500]
        self.y_train = self.y_train[0:500]
        self.y_valid = self.y_valid[0:500]
        self.y_test = self.y_test[0:500]

    def resize(self, img, shape):
        img_resized = []
        for num in range(0, img.shape[0]):
              img_resized.append(list(cv2.resize(img[num], shape)))
        img_resized = np.array(img_resized)
        return img_resized

    def gray2color(self, img):
        img_color = []
        for num in range(0, img.shape[0]):
              img_color.append(list(cv2.cvtColor(img[num], cv2.COLOR_GRAY2RGB)))
        img_color = np.array(img_color)
        return img_color
    

    def get_image_bbox(self, train_dir, annotation_dir, validation_dir=None, test_dir=None,
                       batch_size=10,
                       test_batch_size=1,
                       resize_shape=(255, 255),
                       rescale=None,
                       shuffle=True):
        '''
        指定したディレクトリ内にある画像とBBOX情報を取得するジェネレータを作成する

        Paramteres
        --------------
        train_dir : str
            訓練データのディレクトリへのパス
        annotation_dir : str
            Annotationのディレクトリへのパス
        validation_dir : str
            検証データのディレクトリへのパス
        test_dir : str
            テストデータのディレクトリへのパス
        batch_size : int
            訓練データ（と検証データ）のバッチサイズ
        test_batch_size : int
            テストデータのバッチサイズ（デフォルトは1）
        resize_shape : tuple
            画像データをこのサイズにリサイズする
        rescale : float
            画素値のリスケーリング係数。Noneか0ならば適用しない
        shuffle : bool
            データをシャッフルするかどうか
        
        Notes
        --------------
        ディレクトリ構造は以下の通り。
        まず画像ファイルを読み込み、同じファイル名（拡張子以外）をAnnotationディレクトリから探す
        AnnotationはXMLファイルで固定
        ※kerasのdatageneratorを利用するため、画像のディレクトリ構造に無駄がある。クラスはAnnotationデータのものを使用
        train/
            ┣ class1
            ┃   ┣ class1_01.jpg
            ┃   ┗ class1_02.jpg
            ...
        validation/
            ┣ class1
            ┃   ┣ class1_11.jpg
            ...
        test/
            ┣ class1
            ┃   ┣ class1_12.jpg
            ...
        Annotation/
            ┣ class1_01.xml
            ┣ class1_02.xml
        '''
        # 画像データの読み込み
        self.get_user_data(train_dir, validation_dir=validation_dir, test_dir=test_dir,
                           batch_size=batch_size,   # 一旦バッチ数を1とする
                           test_batch_size=1,
                           resize_shape=resize_shape,
                           rescale=rescale,
                           horizontal_flip=False,  # augumentationするとBBOX情報の変換が難しくなる。augumentationやるならデータ読み込みから自作
                           shuffle=False)  # 画像とBBOXの紐づけのためここではシャッフルしない
        
        # 画像データと画像ファイル名を対応づける
        self.x_train, self.image_filename_train = self._get_image_filename_from_datagenrator(self.train_generator)
        if validation_dir is not None:
            self.x_valid, self.image_filename_valid = self._get_image_filename_from_datagenrator(self.validation_generator)
        if test_dir is not None:
            self.x_test, self.image_filename_test = self._get_image_filename_from_datagenrator(self.test_generator)
        
        # 画像ファイル名からAnntationファイル名に変換
        self.annotation_filename_train = list(map(self._image_filename_to_annotation_filename, self.image_filename_train))
        if validation_dir is not None:
            self.annotation_filename_valid = list(map(self._image_filename_to_annotation_filename, self.image_filename_valid))
        if test_dir is not None:
            self.annotation_filename_test = list(map(self._image_filename_to_annotation_filename, self.image_filename_test))
        
        # AnnotationファイルからBBOX情報を取り出し、画像データのインデックスと紐づけ
        self.df_bbox_train = list(map(self._get_bbox_info,
                                      [annotation_dir]*len(self.annotation_filename_train),
                                      self.annotation_filename_train,
                                      list(range(len(self.annotation_filename_train)))))
        self.df_bbox_train = pd.concat(self.df_bbox_train).reset_index(drop=True)
        if validation_dir is not None:
            self.df_bbox_valid = list(map(self._get_bbox_info,
                                        [annotation_dir]*len(self.annotation_filename_valid),
                                        self.annotation_filename_valid,
                                        list(range(len(self.annotation_filename_valid)))))
            self.df_bbox_valid = pd.concat(self.df_bbox_valid).reset_index(drop=True)
        if test_dir is not None:
            self.df_bbox_test = list(map(self._get_bbox_info,
                                        [annotation_dir]*len(self.annotation_filename_test),
                                        self.annotation_filename_test,
                                        list(range(len(self.annotation_filename_test)))))
            self.df_bbox_test = pd.concat(self.df_bbox_test).reset_index(drop=True)
        
        # BBOX情報を、画像サイズの変換に合わせる
        # TODO df_bbox_trainの元画像サイズ(width,height)と、この関数の引数resize_shapeを突き合わせる

        # ジェネレータを作成
        
    
    def _get_image_filename_from_datagenrator(self, datagen):
        """
        datageneratorに含まれる画像とそのファイル名を取得
        
        Parameters
        ------------
        datagen
            kerasのImageDataGeneratorから作成したジェネレータ
        
        Returns
        -----------
        images : numpy.array (shape=(n_data, height, width, channel))
            画像データのリスト
        image_name_list : list of str
            読み込まれた画像のファイル名のリスト
            インデックスはimages, classesと対応
        """
        images = []
        image_name_list = []

        batches_per_epoch = datagen.samples // datagen.batch_size + (datagen.samples % datagen.batch_size > 0)
        for i in range(batches_per_epoch):
            batch = next(datagen)
            current_index = ((datagen.batch_index-1) * datagen.batch_size)
            if current_index < 0:
                if datagen.samples % datagen.batch_size > 0:
                    current_index = max(0,datagen.samples - datagen.samples % datagen.batch_size)
                else:
                    current_index = max(0,datagen.samples - datagen.batch_size)
            images.append(batch[0])
            index_array = datagen.index_array[current_index:current_index + datagen.batch_size].tolist()
            image_name_list += [datagen.filenames[idx] for idx in index_array]
        
        images = np.concatenate(images, axis=0)
        return images, image_name_list
    
    def _image_filename_to_annotation_filename(self, image_filename):
        """
        画像ファイル名をAnnotationファイル名に変換
        """
        filename = image_filename.split('/')[-1]  # ファイルパスの場合、ファイル名のみ取得
        filename_remove_extention = filename.split('.')[0]  # 拡張子を取り除いたファイル
        annotation_filename = filename_remove_extention + '.xml'
        return annotation_filename
    
    def _get_bbox_info(self, annotation_dir, annotation_filename, img_index):
        """
        annotationファイルからbbox情報を読み込み。画像のインデックスと紐付ける
        """
        # annotation読み込み
        annotation_filepath = os.path.join(annotation_dir, annotation_filename)
        img_filename, df_bbox_info = xml_to_labels(annotation_filepath)

        # bbox情報に画像ファイル名、画像インデックスを紐付ける
        df_bbox_info.loc[:, 'image_filename'] = [img_filename] * len(df_bbox_info)
        df_bbox_info.loc[:, 'image_index'] = [img_index] * len(df_bbox_info)

        return df_bbox_info