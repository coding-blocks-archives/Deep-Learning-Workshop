{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Deep Learning",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "okyBh_41UPNT",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "import keras\n",
        "from keras.datasets import mnist\n",
        "\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense\n",
        "\n",
        "from keras.utils import to_categorical\n",
        "\n",
        "from keras.preprocessing import image"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VGVhNO3DWFiL",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "d137628d-999d-4449-e326-f01bc90e8f63"
      },
      "source": [
        "(x_train, y_train), (x_test, y_test) = mnist.load_data()"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz\n",
            "11493376/11490434 [==============================] - 0s 0us/step\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xtWIJ850Wgp0",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "b2f1eab7-07ba-4a11-bb1b-85b7859c1804"
      },
      "source": [
        "x_train.shape"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 28, 28)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "K9vl5uPiWyun",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "14f80700-d3c7-4c7b-fb7c-4eb49fc9af4a"
      },
      "source": [
        "x_test.shape"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000, 28, 28)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HfUVecInW8DS",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "95cddbbc-20c0-4f9b-c36d-21a74da57db4"
      },
      "source": [
        "y_train.shape"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000,)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8_9SFE_ZXKlP",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "63c15153-5274-4898-d1f0-13693faa3eb1"
      },
      "source": [
        "y_test.shape"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000,)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VbcX1otzXVGd",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def plot_img(img):\n",
        "  plt.imshow(img.reshape(28,28), cmap=\"gray\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WI-2-JTAXlLX",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "fd15b129-9d39-4d89-d24e-697a7765c9d5"
      },
      "source": [
        "plot_img(x_train[1002])\n",
        "print(\"Image is : \" , y_train[1002])"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Image is :  1\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAALWklEQVR4nO3dXYhc5R3H8d/PvHihXmwqXZcYGiu5KYXEEkKhIViDkubC6I0YpKRUWC8UFAptsBcKpRDa2l4KGwwmxfqCLxilVtMgTYsQskoSk1hNqgkmrAkmF4lXNvrvxZzIGnfObOacM2fS//cDw5x5ntk5fw755TlvM48jQgD+/13RdgEABoOwA0kQdiAJwg4kQdiBJOYOcmW2OfUPNCwiPFN7pZHd9hrb79s+Yntjlc8C0Cz3e53d9hxJH0i6VdJxSXskrY+IQyV/w8gONKyJkX2FpCMR8WFEfC7pGUnrKnwegAZVCftCSR9Pe328aPsa2+O2J21PVlgXgIoaP0EXEROSJiR244E2VRnZT0haNO319UUbgCFUJex7JC2xfYPt+ZLulrS9nrIA1K3v3fiIOG/7AUmvS5ojaUtEHKytMgC16vvSW18r45gdaFwjN9UAuHwQdiAJwg4kQdiBJAg7kARhB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkCDuQBGEHkiDsQBKEHUiCsANJEHYgCcIOJEHYgSQIO5DEQKdsxuXn+eefL+0fGRkp7V+9enWd5aACRnYgCcIOJEHYgSQIO5AEYQeSIOxAEoQdSILr7Ch1++23l/Zv3rx5QJWgqkpht31U0jlJX0g6HxHL6ygKQP3qGNl/HBGf1vA5ABrEMTuQRNWwh6Q3bL9te3ymN9getz1pe7LiugBUUHU3fmVEnLD9bUk7bP87InZNf0NETEiakCTbUXF9APpUaWSPiBPF8ylJL0laUUdRAOrXd9htX2X7mgvLkm6TdKCuwgDUq8pu/Kikl2xf+Jy/RMTfaqkKQO36DntEfChpaY21AGgQl96AJAg7kARhB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkCDuQBGEHkiDsQBKEHUiCsANJEHYgCcIOJEHYgSQIO5AEYQeSIOxAEoQdSIKwA0kQdiAJwg4kQdiBJAg7kARhB5Ig7EAShB1IgrADSfQMu+0ttk/ZPjCtbYHtHbYPF88jzZYJoKrZjOxPSlpzUdtGSTsjYomkncVrAEOsZ9gjYpekMxc1r5O0tVjeKumOmusCULO5ff7daERMFcufSBrt9kbb45LG+1wPgJr0G/avRETYjpL+CUkTklT2PgDN6vds/EnbY5JUPJ+qryQATeg37NslbSiWN0h6uZ5yADSl52687acl3SzpWtvHJT0iaZOk52zfK+mYpLuaLBLNueeee0r7584t/yeybdu2OstBg3qGPSLWd+laXXMtABrEHXRAEoQdSIKwA0kQdiAJwg4kUfkOOlze5s2bV9pvu7R/8eLFpf27d+++1JLQEEZ2IAnCDiRB2IEkCDuQBGEHkiDsQBKEHUiC6+zJnT59urQ/ovzHhVatWlXa/+yzz15yTWgGIzuQBGEHkiDsQBKEHUiCsANJEHYgCcIOJMF19uReeeWV0v7z588PqBI0jZEdSIKwA0kQdiAJwg4kQdiBJAg7kARhB5Ig7EASPcNue4vtU7YPTGt71PYJ23uLx9pmywRQ1WxG9iclrZmh/U8Rsax4/LXesgDUrWfYI2KXpDMDqAVAg6ocsz9ge3+xmz/S7U22x21P2p6ssC4AFfUb9scl3ShpmaQpSY91e2NETETE8ohY3ue6ANSgr7BHxMmI+CIivpS0WdKKessCULe+wm57bNrLOyUd6PZeAMOh5/fZbT8t6WZJ19o+LukRSTfbXiYpJB2VdF+DNWKIzZ8/v+0SMEs9wx4R62dofqKBWgA0iDvogCQIO5AEYQeSIOxAEoQdSMK9puStdWX24FaGWrz22mul/bfccktp/5VXXllnOZiFiPBM7YzsQBKEHUiCsANJEHYgCcIOJEHYgSQIO5AEUzaj1EcffVTaP2fOnNL+pUuXdu3bt29fXzWhP4zsQBKEHUiCsANJEHYgCcIOJEHYgSQIO5AE19lRyRVXlI8Xy5Yt69rHdfbBYmQHkiDsQBKEHUiCsANJEHYgCcIOJEHYgSS4zo5GnT17tu0SUOg5stteZPtN24dsH7T9YNG+wPYO24eL55HmywXQr9nsxp+X9IuI+J6kH0q63/b3JG2UtDMilkjaWbwGMKR6hj0ipiLinWL5nKT3JC2UtE7S1uJtWyXd0VSRAKq7pGN224sl3SRpt6TRiJgquj6RNNrlb8YljfdfIoA6zPpsvO2rJb0g6aGI+NpZl+jMDjnjpI0RMRERyyNieaVKAVQyq7DbnqdO0J+KiBeL5pO2x4r+MUmnmikRQB16Ttls2+ock5+JiIemtf9e0umI2GR7o6QFEfHLHp/FlM2XmWPHjpX2X3fddaX9TNk8eN2mbJ7NMfuPJP1U0ru29xZtD0vaJOk52/dKOibprjoKBdCMnmGPiH9JmvF/Ckmr6y0HQFO4XRZIgrADSRB2IAnCDiRB2IEk+IorSo2NjZX2v/XWWwOqBFUxsgNJEHYgCcIOJEHYgSQIO5AEYQeSIOxAElxnRyUHDx5suwTMEiM7kARhB5Ig7EAShB1IgrADSRB2IAnCDiTR83fja10ZvxsPNK7b78YzsgNJEHYgCcIOJEHYgSQIO5AEYQeSIOxAEj3DbnuR7TdtH7J90PaDRfujtk/Y3ls81jZfLoB+9bypxvaYpLGIeMf2NZLelnSHOvOxfxYRf5j1yripBmhct5tqZjM/+5SkqWL5nO33JC2stzwATbukY3bbiyXdJGl30fSA7f22t9ge6fI347YnbU9WqhRAJbO+N9721ZL+Iem3EfGi7VFJn0oKSb9RZ1f/5z0+g914oGHdduNnFXbb8yS9Kun1iPjjDP2LJb0aEd/v8TmEHWhY31+EsW1JT0h6b3rQixN3F9wp6UDVIgE0ZzZn41dK+qekdyV9WTQ/LGm9pGXq7MYflXRfcTKv7LMY2YGGVdqNrwthB5rH99mB5Ag7kARhB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkCDuQBGEHkiDsQBKEHUiCsANJ9PzByZp9KunYtNfXFm3DaFhrG9a6JGrrV521fadbx0C/z/6NlduTEbG8tQJKDGttw1qXRG39GlRt7MYDSRB2IIm2wz7R8vrLDGttw1qXRG39GkhtrR6zAxictkd2AANC2IEkWgm77TW237d9xPbGNmroxvZR2+8W01C3Oj9dMYfeKdsHprUtsL3D9uHiecY59lqqbSim8S6ZZrzVbdf29OcDP2a3PUfSB5JulXRc0h5J6yPi0EAL6cL2UUnLI6L1GzBsr5L0maRtF6bWsv07SWciYlPxH+VIRPxqSGp7VJc4jXdDtXWbZvxnanHb1Tn9eT/aGNlXSDoSER9GxOeSnpG0roU6hl5E7JJ05qLmdZK2Fstb1fnHMnBdahsKETEVEe8Uy+ckXZhmvNVtV1LXQLQR9oWSPp72+riGa773kPSG7bdtj7ddzAxGp02z9Ymk0TaLmUHPabwH6aJpxodm2/Uz/XlVnKD7ppUR8QNJP5F0f7G7OpSicww2TNdOH5d0ozpzAE5JeqzNYoppxl+Q9FBEnJ3e1+a2m6GugWy3NsJ+QtKiaa+vL9qGQkScKJ5PSXpJncOOYXLywgy6xfOpluv5SkScjIgvIuJLSZvV4rYrphl/QdJTEfFi0dz6tpuprkFttzbCvkfSEts32J4v6W5J21uo4xtsX1WcOJHtqyTdpuGbinq7pA3F8gZJL7dYy9cMyzTe3aYZV8vbrvXpzyNi4A9Ja9U5I/8fSb9uo4YudX1X0r7icbDt2iQ9rc5u3X/VObdxr6RvSdop6bCkv0taMES1/Vmdqb33qxOssZZqW6nOLvp+SXuLx9q2t11JXQPZbtwuCyTBCTogCcIOJEHYgSQIO5AEYQeSIOxAEoQdSOJ/rQ2KBucbFEsAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yLwcfDJkYvew",
        "colab_type": "text"
      },
      "source": [
        "# Model Building"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "N989yBJ7Xx35",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "6d89f0d4-3fbf-436d-ed98-707a084991a4"
      },
      "source": [
        "x_train.shape"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 28, 28)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tb9Qv-TuZ-BV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x_train = x_train.reshape(60000, 784)\n",
        "x_test = x_test.reshape(10000, 784)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G6Sqf5haac_V",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "584666ad-6613-4eba-c32d-0356f6d11dfe"
      },
      "source": [
        "x_train.shape"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 784)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n3oMghmubT7E",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "267d7d26-5377-4eb8-c3e2-db093945ca01"
      },
      "source": [
        "x_test.shape"
      ],
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000, 784)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8n75LZSYbWik",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model = Sequential()\n",
        "model.add( Dense(units=32, activation='relu', input_shape = (784,))  ) # input_shape only for first layer\n",
        "model.add( Dense(units=64, activation='relu' ))\n",
        "model.add( Dense(units=128, activation = 'relu'))\n",
        "model.add( Dense(units=32, activation = 'relu'))\n",
        "model.add( Dense(units=10, activation='softmax')) # final  softmax = > probabilities"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-lebpz3JeOj5",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 323
        },
        "outputId": "42e49a4c-5c0a-4358-fbe3-4ab2ec70dff0"
      },
      "source": [
        "model.summary()"
      ],
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"sequential_1\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "dense_1 (Dense)              (None, 32)                25120     \n",
            "_________________________________________________________________\n",
            "dense_2 (Dense)              (None, 64)                2112      \n",
            "_________________________________________________________________\n",
            "dense_3 (Dense)              (None, 128)               8320      \n",
            "_________________________________________________________________\n",
            "dense_4 (Dense)              (None, 32)                4128      \n",
            "_________________________________________________________________\n",
            "dense_5 (Dense)              (None, 10)                330       \n",
            "=================================================================\n",
            "Total params: 40,010\n",
            "Trainable params: 40,010\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nEHxZ8sEeQGL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# \"adam\"/\"sgd\"/\"rmsprop\"\n",
        "model.compile(optimizer=\"adam\", loss=\"categorical_crossentropy\", metrics=['accuracy'])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ULg4_pTyfrOg",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "10a3ad8c-f0f4-42a0-9b92-29794b81abbf"
      },
      "source": [
        "y_train.shape"
      ],
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000,)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 38
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hc_sY2edfsw5",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_train = to_categorical(y_train)\n",
        "y_test = to_categorical(y_test)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kyXLd42ofsoX",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "74959739-d74b-44b3-ab8b-262dfcfea43e"
      },
      "source": [
        "y_train.shape"
      ],
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 10)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ky7fzBPSfseZ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "76b65ca4-a52f-49f8-85d2-4b8d5a56268f"
      },
      "source": [
        "y_test.shape"
      ],
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000, 10)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LYs7156nfsBQ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 394
        },
        "outputId": "bcc30786-5937-457c-f828-7ad68d9b0ac7"
      },
      "source": [
        "hist = model.fit(x=x_train, y= y_train, batch_size=32,epochs = 10, validation_data=(x_test, y_test) )"
      ],
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train on 60000 samples, validate on 10000 samples\n",
            "Epoch 1/10\n",
            "60000/60000 [==============================] - 4s 65us/step - loss: 0.6172 - accuracy: 0.8299 - val_loss: 0.3001 - val_accuracy: 0.9175\n",
            "Epoch 2/10\n",
            "60000/60000 [==============================] - 3s 57us/step - loss: 0.2454 - accuracy: 0.9300 - val_loss: 0.2149 - val_accuracy: 0.9381\n",
            "Epoch 3/10\n",
            "60000/60000 [==============================] - 3s 53us/step - loss: 0.1899 - accuracy: 0.9456 - val_loss: 0.2000 - val_accuracy: 0.9437\n",
            "Epoch 4/10\n",
            "60000/60000 [==============================] - 3s 53us/step - loss: 0.1624 - accuracy: 0.9536 - val_loss: 0.1737 - val_accuracy: 0.9545\n",
            "Epoch 5/10\n",
            "60000/60000 [==============================] - 3s 54us/step - loss: 0.1443 - accuracy: 0.9589 - val_loss: 0.1554 - val_accuracy: 0.9592\n",
            "Epoch 6/10\n",
            "60000/60000 [==============================] - 3s 57us/step - loss: 0.1264 - accuracy: 0.9635 - val_loss: 0.1566 - val_accuracy: 0.9588\n",
            "Epoch 7/10\n",
            "60000/60000 [==============================] - 3s 55us/step - loss: 0.1198 - accuracy: 0.9659 - val_loss: 0.1384 - val_accuracy: 0.9627\n",
            "Epoch 8/10\n",
            "60000/60000 [==============================] - 3s 54us/step - loss: 0.1122 - accuracy: 0.9682 - val_loss: 0.1514 - val_accuracy: 0.9571\n",
            "Epoch 9/10\n",
            "60000/60000 [==============================] - 3s 54us/step - loss: 0.1024 - accuracy: 0.9702 - val_loss: 0.1501 - val_accuracy: 0.9610\n",
            "Epoch 10/10\n",
            "60000/60000 [==============================] - 3s 54us/step - loss: 0.0966 - accuracy: 0.9728 - val_loss: 0.1755 - val_accuracy: 0.9622\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QMq3qwm5hKzw",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "88fb769a-0ff9-4b15-8bff-7407b2212516"
      },
      "source": [
        "model.evaluate(x_test, y_test)"
      ],
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "10000/10000 [==============================] - 0s 23us/step\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[0.17551743202386424, 0.9621999859809875]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rspCgHcCh1G7",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "bd8b9c32-2c25-4890-9f2b-ad6d80f6086a"
      },
      "source": [
        "plot_img(x_test[5001])"
      ],
      "execution_count": 59,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NeSOYmDdiDrr",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_pred = model.predict_classes(x_test)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "640IoZH5iLnD",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "7acb9725-3f75-4c1b-e0e1-59e82480372a"
      },
      "source": [
        "y_pred[5001]"
      ],
      "execution_count": 61,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "9"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rLUSSqAPiZNE",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "51aa358e-891d-4da7-ece8-5360d1935e81"
      },
      "source": [
        "model.predict_classes(x_test[[5000]])"
      ],
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([3])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 63
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mpgPJ0pvi63Y",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 71
        },
        "outputId": "44c60d29-a511-4efb-ee44-405e74116498"
      },
      "source": [
        "# this code is custom image. \n",
        "\n",
        "img = image.load_img(\"untitled.png\", grayscale=True, target_size=(28,28))\n",
        "img = np.array(img)\n",
        "img = img.reshape(1, 784)\n",
        "\n",
        "prediction = model.predict_classes(img)"
      ],
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/utils.py:107: UserWarning: grayscale is deprecated. Please use color_mode = \"grayscale\"\n",
            "  warnings.warn('grayscale is deprecated. Please use '\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0z-vScsBj0qZ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "84f399b9-05d0-45f8-b7d3-50d9a76b6ac4"
      },
      "source": [
        "prediction"
      ],
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7Hj6_Sgwj3Em",
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