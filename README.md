# Issueから調べたこと
① https://github.com/tensorflow/models/issues/9133
・trainする時はpbtxtとconfigのnum_classが一緒である必要あり
・evaluateする時はpbtxtとconfigとckptのnum_classが一緒である必要あり

② https://github.com/tensorflow/models/issues/8892#issuecomment-662164279
・efficientdetを遣う時num_classをデフォルト90から変える時は，fine_tune_checkpoint_type to "detection"にする必要がある．
・

# Tensorflow公式colab_tutorials

https://github.com/tensorflow/models/tree/master/research/object_detection/colab_tutorials

# tfのファイル
checkpoint : 重み
frozen_inference_graph.pb : 
model.ckpt.data-00000-of-00001 : 
model.ckpt.index : 
model.ckpt.meta : 構造？？
pipeline.config :
saved_model :

# 参考
https://qiita.com/sanoyo/items/1a5c4e8671203d190fca

https://qiita.com/t_shimmura/items/1ebd2414310f827ed608

https://qiita.com/cfiken/items/bcdd7eb945c5c3b2bb5f
