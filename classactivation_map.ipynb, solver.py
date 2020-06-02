{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "classactivation_map.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.6.9"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ex-hasuolab/class-activation-map/blob/feature%2F%238%2F%2312%2F20200602%2Fchopprin/classactivation_map.ipynb%2C%20solver.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VKBwOsdPcGTP",
        "colab_type": "code",
        "outputId": "3f4d8daf-367f-401c-d289-cb28e6b28b15",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 55
        }
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JZ94RrBImdik",
        "colab_type": "code",
        "outputId": "fcf85dc9-ea10-4100-c471-9217a0132323",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 71
        }
      },
      "source": [
        "%tensorflow_version 1.x\n",
        "import tensorflow as tf\n",
        "print(tf.__version__)\n",
        "!pip install -U efficientnet==0.0.4"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "TensorFlow 1.x selected.\n",
            "1.15.2\n",
            "Requirement already up-to-date: efficientnet==0.0.4 in /usr/local/lib/python3.6/dist-packages (0.0.4)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c-KJ0z47iDmh",
        "colab_type": "code",
        "outputId": "6fac7c57-6f44-4757-a4dc-be3da1612db0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 91
        }
      },
      "source": [
        "# Dataloader ： utils/loder.pyに記述するべき内容\n",
        "%load_ext autoreload\n",
        "import numpy as np\n",
        "import cv2\n",
        "from math import ceil\n",
        "from scipy import ndimage\n",
        "from sklearn.model_selection import train_test_split\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.image as mpimg\n",
        "import seaborn as sns\n",
        "%matplotlib inline\n",
        "from tensorflow.keras import utils\n",
        "import keras\n",
        "#from tensorflow.keras.datasets.cifar10 import load_data\n",
        "from tensorflow.keras.datasets.mnist import load_data\n"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.\n",
            "  import pandas.util.testing as tm\n",
            "Using TensorFlow backend.\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GjJYniLZxfft",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%autoreload"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nZaBG3BijGH0",
        "colab_type": "code",
        "outputId": "1bb7b64a-61d8-46da-e264-d3175c86030e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 53
        }
      },
      "source": [
        "from dataloader import Dataloader\n",
        "dataloader_ins = Dataloader()\n",
        "dataloader_ins.get_data(resize_mode = True, resize_shape = (56,56), cvtColor_mode = True)\n",
        "\n",
        "print(\"x_train shape : {}\".format(dataloader_ins.x_train.shape))\n",
        "print(\"y_train shape : {}\".format(dataloader_ins.y_train.shape))"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "x_train shape : (500, 56, 56, 3)\n",
            "y_train shape : (500, 10)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M0CVZ-l4mDkL",
        "colab_type": "code",
        "outputId": "d219efc1-2734-4900-917f-f9d670dc519f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 89
        }
      },
      "source": [
        "from usermodel_efficientnet import Usermodel_efficientnet\n",
        "\n",
        "data_num = dataloader_ins.x_train.shape[0]\n",
        "cut_size = {\n",
        "    \"height\" : dataloader_ins.x_train.shape[1],\n",
        "    \"width\" : dataloader_ins.x_train.shape[2]\n",
        "}\n",
        "channel = dataloader_ins.x_train.shape[3]\n",
        "category_count = dataloader_ins.y_train.shape[1]\n",
        "\n",
        "print(data_num)\n",
        "print(cut_size)\n",
        "print(channel)\n",
        "print(category_count)\n",
        "\n",
        "usermodel_ins = Usermodel_efficientnet(cut_size, channel, category_count)"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "500\n",
            "{'height': 56, 'width': 56}\n",
            "3\n",
            "10\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PXy6GS3q5-s2",
        "colab_type": "code",
        "outputId": "8d68a8cd-dcde-4e05-c1e3-441b8da567b4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 289
        }
      },
      "source": [
        "from solver import Solver\n",
        "solver_ins = Solver(dataloader_ins, usermodel_ins, batch_size = 10, n_epochs = 5)\n",
        "model = solver_ins.get_model()"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "input_shape : (56, 56, 3)\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/efficientnet/initializers.py:44: The name tf.random_normal is deprecated. Please use tf.random.normal instead.\n",
            "\n",
            "WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "If using Keras pass *_constraint arguments to layers.\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/efficientnet/layers.py:38: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/efficientnet/layers.py:42: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Deprecated in favor of operator or tf.math.divide.\n",
            "WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use tf.where in 2.0, which has the same broadcast rule as np.where\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qtH23JC58PqL",
        "colab_type": "code",
        "outputId": "ef938a03-3d6c-4ed1-8341-9f8d1c4d6540",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 271
        }
      },
      "source": [
        "solver_ins.train(model)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "x_train shape (500, 56, 56, 3)\n",
            "y_train shape (500, 10)\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.\n",
            "\n",
            "Train on 500 samples, validate on 500 samples\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/callbacks/tensorboard_v1.py:198: The name tf.summary.histogram is deprecated. Please use tf.compat.v1.summary.histogram instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/callbacks/tensorboard_v1.py:200: The name tf.summary.merge_all is deprecated. Please use tf.compat.v1.summary.merge_all instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/callbacks/tensorboard_v1.py:203: The name tf.summary.FileWriter is deprecated. Please use tf.compat.v1.summary.FileWriter instead.\n",
            "\n",
            "Epoch 1/5\n",
            "290/500 [================>.............] - ETA: 12s - loss: 0.2727 - categorical_accuracy: 0.5138"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HucPP26S042w",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%load_ext tensorboard"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SRVr1UOfsXco",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%tensorboard --logdir=\"./log\"\n",
        "# ローカルで見たい時はlocalhost:6006にアクセスする事"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cb61UD-98sod",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}