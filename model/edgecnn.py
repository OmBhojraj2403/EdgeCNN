"""
Implementation of EdgeCNN in Tensorflow
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras import backend
from keras.engine import training
from keras.utils import layer_utils
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    GlobalMaxPooling2D, BatchNormalization, AveragePooling2D, Input
from tensorflow.keras.regularizers import l2

def edge_block(x, blocks, growth_rate, name):
    """An edge block.

    Args:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = edge_conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x


def edge_conv_block(x, growth_rate, name):
    """A building block for an edge block.

    Args:
      x: input tensor.
      growth_rate: float, growth rate at edge layers.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    x1 = Conv2D(4 * growth_rate, 3, padding="same",
                use_bias=False, name=name + '_1_conv',
                kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4))(x)
    x1 = BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(x1)
    x1 = Activation('relu', name=name + '_0_relu')(x1)

    x1 = Conv2D(growth_rate, 3, padding='same',
                use_bias=False, name=name + '_2_conv',
                kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4))(x1)
    x1 = BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x = Concatenate(name=name + '_concat')([x, x1])
    return x


def transition_block(x, name):
    """A transition block.

    Args:
      x: input tensor.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    x = AveragePooling2D(2, strides=2, name=name + '_avg_pool')(x)
    return x


def classifier(x, classes, classifier_activation, name):
    """The classification block.

    Args:
      x: input tensor.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    x = GlobalAveragePooling2D(name='glob_avg_pool')(x)
    x = Dense(classes, activation=classifier_activation, name='predictions',
              kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4))(x)
    return x


def EdgeCNN(classes, input_shape, growth_rate=4, blocks=[4, 4, 7], block_config=[4, 4, 4], include_top=True, weights=None, input_tensor=None, pooling=None, classifier_activation='softmax'):
    """Instantiates the EdgeCNN architecture.

    Reference:
    - [EdgeCNN: Convolutional Neural Network Classification
  Model with small inputs for Edge Computing](
        https://arxiv.org/pdf/1909.13522) (CVPR 2017)

    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.

    Note: each Keras Application expects a specific kind of input preprocessing.
    For DenseNet, call `tf.keras.applications.densenet.preprocess_input` on your
    inputs before passing them to the model.
    `densenet.preprocess_input` will scale pixels between 0 and 1 and then
    will normalize each channel with respect to the ImageNet dataset statistics.

    Args:
      classes: number of classes to classify images into
      input_shape: shape tuple, only to be specified
        (with `'channels_last'` data format)
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
      growth_rate: rate of growth of kernels in each edge block
      blocks: numbers of building blocks for the three edge layers.
      block_config: []
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

#   x = ZeroPadding2D(padding=((1, 1), (1, 1)))(img_input)
    x = Conv2D(32, 3, padding="same", use_bias=True,
               name='conv1', kernel_regularizer=l2(5e-4), 
               bias_regularizer=l2(5e-4))(img_input)
    x = MaxPooling2D(3, strides=2, padding="same", name='pool1')(x)

    x = edge_block(x, blocks[0], growth_rate=growth_rate *
                   block_config[0], name='conv2')
    x = transition_block(x, name='pool2')
    x = edge_block(x, blocks[1], growth_rate=growth_rate *
                   block_config[1], name='conv3')
    x = transition_block(x, name='pool3')
    x = edge_block(x, blocks[2], growth_rate=growth_rate *
                   block_config[2], name='conv4')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)

    if include_top:
        x = classifier(x, classes, classifier_activation, name="predictions")
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name='edgecnn')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

# model = EdgeCNN(classes=2, input_shape=(44, 44, 3), growth_rate=8)
# model.summary()


# class EdgeBlock(tf.keras.layers.Layer):

#     def __init__(self, growth_rate, num):
#         super(EdgeBlock, self).__init__()
#         self.growth_rate = growth_rate
#         self.num = num

#     def build(self, input_shape):
#         # self.features = keras.models.Sequential()
#         self.conv1 = Conv2D(filters=4*self.growth_rate, kernel_size=3, padding="valid", name="conv1_"+self.num)
#         self.bn1 = BatchNormalization(name="bn1"+self.num)
#         self.relu1 = ReLU(name="relu1"+self.num)
#         self.conv2 = Conv2D(filters=4*self.growth_rate, kernel_size=3, padding="valid", name="conv2"+self.num)
#         self.bn2 = BatchNormalization(name="bn2"+self.num)
#         self.relu2 = ReLU(name="relu2"+self.num)
#         super(EdgeBlock, self).build(input_shape)  # Identical to self.built = True

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         return x

#     # # Need to define get_config method in order to sequentialize the model constructed from the customized Layer by Functional API.
#     # def get_config(self):
#     #     config = super(EdgeBlock, self).get_config()
#     #     config.update({'kernel_size': self.kernel_size})
#     #     return config


# class Transition_Layer(tf.keras.layers.Layer):

#     def __init__(self, num):
#         super(Transition_Layer, self).__init__()
#         self.num = num

#     def build(self, input_shape):
#         self.avg_pool = AveragePooling2D(pool_size=2, strides=2, name="avg_pool"+self.num)
#         super(Transition_Layer, self).build(input_shape)  # Identical to self.built = True

#     def call(self, inputs):
#         x = self.avg_pool(inputs)
#         return x


# class Classification_Layer(tf.keras.layers.Layer):

#     def __init__(self):
#         super(Classification_Layer, self).__init__()

#     def build(self, input_shape):
#         self.glob_avg_pool = GlobalAveragePooling2D(name = "glob_avg_pool")
#         self.dense = Dense(152, activation="softmax", name= "dense152")
#         super(Classification_Layer, self).build(input_shape)  # Identical to self.built = True

#     def call(self, inputs):
#         x = self.glob_avg_pool(inputs)
#         x = self.dense(x)
#         return x

# class EdgeCNN(keras.models.Model):
#     def __init__(self, growth_rate):
#         super(EdgeCNN, self).__init__()
#         self.growth_rate = growth_rate

#     def build(self, input_shape):

#         self.conv0 = Conv2D(filters=32, kernel_size=3, padding="same", use_bias=True, name="conv0")
#         self.pool0 = MaxPooling2D(pool_size=3, strides=2)
#         self.edgeblock1 = EdgeBlock(growth_rate=self.growth_rate, num="_1")
#         self.transition1 = Transition_Layer(num="_1")
#         self.edgeblock2 = EdgeBlock(growth_rate=self.growth_rate, num="_2")
#         self.transition2 = Transition_Layer(num="_2")
#         self.edgeblock3 = EdgeBlock(growth_rate=self.growth_rate, num="_3")
#         self.classification = Classification_Layer()
#         super(EdgeCNN,self).build(input_shape)

#     def call(self, inputs):
#         x = self.conv0(inputs)
#         x = self.pool0(x)
#         x = self.edgeblock1(x)
#         x = self.transition1(x)
#         x = self.edgeblock2(x)
#         x = self.transition2(x)
#         x = self.edgeblock3(x)
#         x = self.classification(x)
#         return(x)

# tf.keras.backend.clear_session()
# model = EdgeCNN(growth_rate=8)
# model.build(input_shape =(1,44,44,3))
# model.summary()
# model.compile(optimizer='Nadam',
#             loss='binary_crossentropy',
#             metrics=['accuracy',"AUC"])

# from keras.utils import plot_model
# tf.keras.utils.plot_model(model, to_file='model.png')
