from gan.Interface import *

class Generator(Base):
    def __init__(self):
        super(Generator, self).__init__()

        dropout = self._dropout
        depth = self._depth * 4
        dim = 7

        self.structure.add(Dense(dim * dim * depth, input_dim=100))
        self.structure.add(BatchNormalization(momentum=0.9))
        self.structure.add(Activation('relu'))

        self.structure.add(Reshape((dim, dim, depth)))
        self.structure.add(Dropout(dropout))

        self.structure.add(UpSampling2D())
        self.structure.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        self.structure.add(BatchNormalization(momentum=0.9))
        self.structure.add(Activation('relu'))

        self.structure.add(UpSampling2D())
        self.structure.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        self.structure.add(BatchNormalization(momentum=0.9))
        self.structure.add(Activation('relu'))

        self.structure.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        self.structure.add(BatchNormalization(momentum=0.9))
        self.structure.add(Activation('relu'))

        self.structure.add(Conv2DTranspose(1, 5, padding='same'))
        self.structure.add(Activation('sigmoid'))