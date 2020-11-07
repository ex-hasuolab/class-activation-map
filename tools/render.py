from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
import numpy as np


def plot_detections(image_np,
                    boxes,
                    classes,
                    index_to_label,
                    scores=None,
                    figsize=(12, 16),
                    image_name=None):
    """
    画像、BBOX、スコアをプロット

    Args:
    image_np : np.array, shape=(img_height, img_width, 3), dtype=int
        画像データ（必ずintにする）
    boxes : np.array, shape (N, 4)
        BBOX（x, y, w, h）
    classes: np.array, shape (N,)
        クラスインデックス（0始まり）
        （内部で1始まりのインデックスに変換してからvisualize_boxes_and_labels_on_image_array関数に入れている）
    index_to_label: dict
        クラスインデックスとクラス名の対応関係。{index: label}
    scores: np.array, shape (N,) or None
        If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
    figsize: 画像サイズ
    image_name: 保存する場合は画像ファイル名
    """
    image_np_with_annotations = image_np.copy()
    classes_one_oriented = np.array([index+1 for index in classes])  # 1始まりのクラスインデックスに変換
    category_index = {index+1: {'id': index+1, 'name': label} for index, label in index_to_label.items()}  # 1始まりのインデックス
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes_one_oriented,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)
