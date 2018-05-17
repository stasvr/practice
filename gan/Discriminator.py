from gan.Interface import *

class Discriminator(Base):
    def __init__(self):
        super(Discriminator, self).__init__()

        depth = self._depth
        dropout = self._dropout
        input_shape = (28, 28, 1)

        self.structure.add(Conv2D(depth * 1, 5, strides=2, input_shape=input_shape,
                                  padding='same', activation=LeakyReLU(alpha=0.2)))
        self.structure.add(Dropout(dropout))

        self.structure.add(Conv2D(depth * 2, 5, strides=2, padding='same',
                                  activation=LeakyReLU(alpha=0.2)))
        self.structure.add(Dropout(dropout))

        self.structure.add(Conv2D(depth * 4, 5, strides=2, padding='same',
                                  activation=LeakyReLU(alpha=0.2)))
        self.structure.add(Dropout(dropout))

        self.structure.add(Conv2D(depth * 2, 5, strides=1, padding='same',
                                  activation=LeakyReLU(alpha=0.2)))
        self.structure.add(Dropout(dropout))
        self.structure.add(Flatten())

        self.structure.add(Dense(1))
        self.structure.add(Activation('sigmoid'))