# 参考：https://godatadriven.com/blog/keras-multi-label-classification-with-imagedatagenerator/

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import pandas as pd


def xml_to_labels(annotation_file):
    '''
    指定したxmlファイル(JSON)から、画像ファイル名とアノテーションを取得する
    bboxはJSONの第1層のもののみ取得
    （例：object=personの中に hand, foot...等が含まれるデータもある）
    
    Parameters
    --------------
    annotations_file : str (or pathlib.PosixPath)
        xmlファイルパス

    Returns
    --------------
    img_filename : str
        画像ファイル名
    df_annotations : pandas.DataFrame, shape=(num_object, 5)
        画像に写っている物体(num_object個)に対し、下記の5つのカラムを持つDF
        ・x: BBOX左下のx座標
        ・y: BBOX左下のy座標
        ・w: BBOXの幅
        ・h: BBOXの高さ
        ・label: クラスラベル
    '''
    with open(annotation_file) as f:
        xml_data = f.read()  # xmlファイルの内容を読み込む

    # xml操作
    root = ET.XML(xml_data)
    obj_to_int = lambda x: int(x.text)
    df_annotations = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'width', 'height', 'label'])
    for i, child in enumerate(root):
        if child.tag == 'filename':
            img_filename = child.text
        if child.tag == 'size':
            for subchild in child:
                if subchild.tag == 'width':
                    width = int(subchild.text)
                if subchild.tag == 'height':
                    height = int(subchild.text)
        if child.tag == 'object':
            # 各objectにname,bndboxタグが必ず1つのみついていることを想定して値を読み取る
            for subchild in child:
                if subchild.tag == 'name':
                    label = subchild.text
                if subchild.tag == 'bndbox':
                    xmin, ymin, xmax, ymax = tuple(map(obj_to_int, subchild.getchildren()))
            # BBOXの幅と高さを計算
            w = xmax - xmin
            h = ymax - ymin
            # DFに追加
            df_annotations = df_annotations.append(
                {'x': xmin, 'y': ymin, 'w': w, 'h': h, 'width': width, 'height': height, 'label': label},
                ignore_index=True
            )
    return img_filename, df_annotations


def annotations_generator(annotations_dir, class_indices):
    '''
    指定したディレクトリ内にある全xmlファイル(JSON)に対し、
        tuple(画像ファイル名, [画像に含まれるラベルのリスト])
    を、1枚の画像毎に取得するジェネレータ

    Parameters
    ---------------
    annotations_dir : str
        xmlファイルのディレクトリパス
        （例：/path/to/dir）
        xmlは Pascal VOC、YOLO 等が使える形式
    class_indices : dict
        クラスラベルからインデックスへの写像
        Noneの場合は自動で追加

    Yields
    ---------------
    tuple = (str, numpy.array(shape=(num_objects, 5)))
        画像ファイル名、アノテーション
    '''
    annotations_dir = Path(annotations_dir).expanduser()
    for annotation_file in annotations_dir.iterdir():
        img_filename, df_annotations = xml_to_labels(annotation_file)
        classlabel_to_index = lambda l: class_indices[l]
        df_annotations['label'] = df_annotations['label'].map(classlabel_to_index)
        annotations_array = df_annotations.values
        yield img_filename, annotations_array