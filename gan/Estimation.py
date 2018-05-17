from gan.Adversarial import Adversarial
from gan.Discriminator import Discriminator
from keras.optimizers import RMSprop
import numpy as np
from scipy import misc as saver


class Process:

    def __init__(self, data, optimizer=[RMSprop(lr=0.00008, clipvalue=1.0, decay=6e-8),
                                        RMSprop(lr=0.00004, clipvalue=1.0, decay=6e-8)]):
        if len(data.shape) != 4:
            raise ValueError("Dataset dimension shape dose not match 4")

        # shape = (None, Width, Height, Chanels)
        self._data = data

        self._optimizer = [i for i in optimizer]

        self._up = Discriminator()
        self._up.structure.compile(loss='binary_crossentropy',
                                   optimizer=self._optimizer[0],
                                   metrics=['accuracy'])

        self._down = Adversarial()
        self._down.structure.compile(loss='binary_crossentropy',
                                     optimizer=self._optimizer[1],
                                     metrics=['accuracy'])

    def run(self, iterations, batch_size):
        print('\n\n\nStart training ... ')
        for iteration in range(iterations):

            images_train = self._data[np.random.randint(0, self._data.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self._down.structure.layers[0].predict(noise)
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0

            d_loss = self._up.structure.train_on_batch(np.concatenate((images_train, images_fake)), y)

            y = np.ones([batch_size, 1])
            a_loss = self._down.structure.train_on_batch(np.random.uniform(-1.0, 1.0, size=[batch_size, 100]), y)

            print('Iteration: {0}, ALossAcc: {1}, DLossAcc: {2}'.format(iteration, a_loss, d_loss))
            if iteration % 50 == 0:
                saver.imsave('./outfile.jpg', images_fake[0].reshape(28, 28))
