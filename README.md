# Issueから調べたこと
① https://github.com/tensorflow/models/issues/9133
・trainする時はpbtxtとconfigのnum_classが一緒である必要あり
・evaluateする時はpbtxtとconfigとckptのnum_classが一緒である必要あり

② https://github.com/tensorflow/models/issues/8892#issuecomment-662164279
・efficientdetを遣う時num_classをデフォルト90から変える時は，fine_tune_checkpoint_type to "detection"にする必要がある．

③ https://github.com/tensorflow/models/issues/2203#issuecomment-323645873
transfer learningをする時に，既存のfeature_extracterを完全にこていしてしまうと，精度も落ちるしそこまで学習が速いわけでも無い．
全体のネットワークの重みを最適化していく方が良い．

https://github.com/tensorflow/models/issues/2203#issuecomment-361045083
ただ，データ数が少ない時はfeature extracterの重みを固定するのがDLの世界では有効であるとされているので，その結論はおかしいのではという人も

https://github.com/tensorflow/models/issues/2203#issuecomment-383756826
レイヤー名を確認して，どの重みを固定しておきたいかはここでいじる事が出来る

?? そもそもDLの世界ではextracterの重みを固定することに大きな高価はあるのだろうか？？

④ 

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
