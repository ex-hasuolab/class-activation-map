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
from .object_detection import get_annotations
import os
import glob
import tensorflow as tf
from PIL import Image

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
            シアー強度（半時計周り）
        zoom_range : float or [float, float]
            ランダムにズームする範囲。
            リストを与えた場合は [lower, upper]。floatを与えた場合は [lower, upper]=[1-zoom_range, 1+zoom_range] 
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
    

    ####################
    # 物体検出用データローダ（TODO ここまでのコードと完全に分離できるので別ファイルでもいい？）
    ####################
    def get_object_detection_data(self, train_dir, validation_dir=None, test_dir=None,
                                  batch_size=10,
                                  test_batch_size=1,
                                  resize_shape=(255, 255),
                                  add_imagefile_extension=None):
        '''
        ディレクトリ名を指定して画像データとBBOXのジェネレータを生成する

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
        resize_shape : tuple or None
            画像データをこのサイズに統一する。Noneの場合はリサイズしない
        add_imagefile_extension : str or optional
            AnnotationのXMLファイルのfilenameに拡張子が含まれていない場合、ここで指定する。

        Notes
        --------------
        ディレクトリ構造は以下の通りにする
        train_dir/
            ┣ images/
            ┃   ┣ image01.jpg
            ┃   ┃ image02.jpg
            ┃   ...
            ┗ annotation/
                 ┣ annotation01.xml
                 ┃ annotation02.xml
                 ...
        ※validation_dir, test_dirも同様
        '''
        #############
        # Annotationデータフレームの作成
        #############
        # 訓練データ
        self.df_annotation_train = self._get_df_annotation(
            os.path.join(train_dir, 'annotation'),
            add_imagefile_extension=add_imagefile_extension)
        # クラス名とインデックスの対応関係を取得
        df_label_index = self.df_annotation_train[['class_label', 'class_index']].drop_duplicates()
        self.label_to_index = df_label_index.set_index('class_label')['class_index'].to_dict()
        self.index_to_label = df_label_index.set_index('class_index')['class_label'].to_dict()

        # 検証データ
        if validation_dir is not None:
            self.df_annotation_validation = self._get_df_annotation(
                os.path.join(validation_dir, 'annotation'),
                class_index_map=self.classlabel_to_index,
                add_imagefile_extension=add_imagefile_extension)
        
        # テストデータ
        if test_dir is not None:
            self.df_annotation_test = self._get_df_annotation(
                os.path.join(test_dir, 'annotation'),
                class_index_map=self.classlabel_to_index,
                add_imagefile_extension=add_imagefile_extension)
        
        ##########
        # generator作成
        ##########
        # 訓練データ
        self.train_generator = self._image_annotation_generator(
            os.path.join(train_dir, 'images'),
            self.df_annotation_train,
            batch_size,
            resize_shape=resize_shape)
        
        # 検証データ
        if validation_dir is not None:
            self.validation_generator = self._image_annotation_generator(
                os.path.join(validation_dir, 'images'),
                self.df_annotation_validation,
                batch_size,
                resize_shape=resize_shape)
        
        # テストデータ
        if test_dir is not None:   
            self.test_generator = self._image_annotation_generator(
                os.path.join(test_dir, 'images'),
                self.df_annotation_test,
                batch_size,
                resize_shape=resize_shape)

    def _get_df_annotation(self,
                           annotation_path,
                           class_index_map=None,
                           add_imagefile_extension=None):
        """
        Annotationデータフレームを取得する

        Paramteres
        --------------
        annotation_path : str
            Annotationデータ(XML)が格納されたディレクトリへのパス
        class_index_map : dict {str: int}, optional
            Annotationファイルに書かれているクラス名(str)とクラスインデックス(int)を対応させる辞書。
            Noneの場合は自動でクラスインデックスを作成。
        add_imagefile_extension : str or optional
            AnnotationのXMLファイルのfilenameに拡張子が含まれていない場合、ここで指定する。
        
        Returns
        -------------
        df_annotation : pandas.DataFrame
            Annotationデータフレーム
            [列]
            ・xmin_rate, ymin_rate, xmax_rate, ymax_rate: BBOXの四隅の相対座標(0~1)
            ・width, height: 画像ファイルの幅・高さ
            ・class_label: クラスラベル
            ・class_index: クラスインデックス（0から順にふる）
            ・image_filename: 各Annotationに対応する画像ファイル名
        """
        # annotationディレクトリにある全xmlファイル名リストを作成
        annotation_filepath_list = glob.glob(os.path.join(annotation_path, '*.xml'))
        assert len(annotation_filepath_list) != 0, 'Annotationファイルが見つかりませんでした'

        # Annotationデータフレーム読み込み
        df_annotation = get_annotations(annotation_filepath_list, class_index_map=class_index_map,
                                        add_imagefile_extension=add_imagefile_extension)
        
        # BBOX情報を割合に変換
        df_annotation['xmin_rate'] = df_annotation['xmin'] / df_annotation['width']
        df_annotation['ymin_rate'] = df_annotation['ymin'] / df_annotation['height']
        df_annotation['xmax_rate'] = df_annotation['xmax'] / df_annotation['width']
        df_annotation['ymax_rate'] = df_annotation['ymax'] / df_annotation['height']

        # 必要な列だけ取り出す
        df_annotation = df_annotation[['xmin_rate', 'ymin_rate', 'xmax_rate', 'ymax_rate',
                                       'width', 'height', 'class_label', 'class_index', 'image_filename']]
        return df_annotation

    def _image_annotation_generator(self,
                                    image_path,
                                    df_annotation,
                                    batch_size,
                                    resize_shape=(255, 255)):
        '''
        画像データとAnnotationを合わせて取得するジェネレータ

        Paramteres
        --------------
        image_path : str
            画像データが格納されたディレクトリへのパス
        df_annotation : pandas.DataFrame
            Annotationデータフレーム
        batch_size : int
            バッチサイズ
        resize_shape : tuple or None
            画像データをこのサイズにリサイズする。Noneの場合はリサイズしない
        
        Yields
        -------------
        image_tensors : list of tf.Tensor(shape=(1, height, width, 3))
            画像のリスト
        bbox_tensors : list of tf.Tensor(shape=(n_bbox, 4))
            BBOXのリスト [ymin_rate, xmin_rage, ymax_rate, xmax_rate]
        class_tensors : list of tf.Tensor(shape=(n_bbox, n_class))
            クラス(one_hot)のリスト
        '''
        # Annotationデータからユニークな画像ファイルパスを取得
        filename_unique = df_annotation['image_filename'].unique()
        # クラス数を取得
        num_class = df_annotation['class_index'].unique().size
        
        while True:
            # バッチ数分の画像ファイル名リストを取得
            filename_batch = np.random.choice(filename_unique, size=batch_size, replace=False)

            # バッチ毎にデータを取得
            image_tensors = []
            bbox_tensors = []
            class_tensors = []
            for filename in filename_batch:
                # 画像
                with Image.open(os.path.join(image_path, filename)) as image:
                    if resize_shape is not None:
                        image = image.resize(resize_shape)
                    image_array = np.array(image)
                image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
                image_tensors.append(image_tensor)

                # BBOX
                df_annotation_batch = df_annotation.loc[df_annotation['image_filename']==filename]
                bbox_array = df_annotation_batch[['ymin_rate', 'xmin_rate', 'ymax_rate', 'xmax_rate']].values
                bbox_tensor = tf.convert_to_tensor(bbox_array, dtype=tf.float32)
                bbox_tensors.append(bbox_tensor)

                # class
                class_index_array = df_annotation_batch['class_index'].values
                one_hot_tensor = tf.one_hot(class_index_array, num_class)
                class_tensors.append(one_hot_tensor)
            
            yield image_tensors, bbox_tensors, class_tensors
