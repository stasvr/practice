from gan.Estimation import Process
from keras.datasets import mnist

if __name__ == '__main__':
    data = mnist.load_data()
    process = Process(data[0][0].reshape(data[0][0].shape[0], 28, 28, 1))
    process.run(500, batch_size=128)