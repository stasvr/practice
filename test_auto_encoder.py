from auto_encoder.Estimation import Process
from auto_encoder.AutoEncoder import Model
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    data = mnist.load_data()
    buff = MinMaxScaler().fit_transform(data[0][0].reshape(data[0][0].shape[0], 784))
    params = {
        'num_input': 784,
        'num_hidden': [392, 196, 98, 49],
        'batch_normalisation': True
    }
    encoder = Model(params)
    process = Process(buff, {'epochs': 10, 'batch_size': 250})
    process.run(encoder)
