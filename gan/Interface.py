from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, Activation, LeakyReLU
from keras.layers import BatchNormalization, Conv2DTranspose, UpSampling2D, Reshape
from keras.models import Sequential
from keras.optimizers import RMSprop

class Base(object):
    def __init__(self, depth=64, dropout=0.4, image_shape=(28, 28, 1)):
        self.structure = Sequential()
        self._depth = depth
        self._dropout = dropout
        self._input_shape = (i for i in image_shape)

    def replace(self, structure):
        del self.structure
        self.structure = structure