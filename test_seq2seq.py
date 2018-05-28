import numpy as np
from seq_to_seq.Estimation import Process
from seq_to_seq.SeqToSeq import Model

def generate_sequence(lower, upper, n, q):
    seq = np.random.randint(lower, upper, size=(q, np.random.choice(n)))
    return seq

if __name__ == '__main__':
    VOCAB = 15
    PAD = 0
    EOS = 3
    
    data = generate_sequence(2, VOCAB, [3,4], 100)
    base = len(data[0])
    for idx, i in enumerate(data):
        r = base - np.random.randint(2)
        data[idx] = [PAD] * (base-r) + [j for jdx, j in enumerate(i) if jdx < r]
    print('First 2 rows of data :\n', data[:2])
    
    params = {
        'embedding_size': 20,
        'vocab_size': VOCAB,
    }
    seq2seq = Model(hidden_units=[10, 10], params=params)
    process = Process(data, {'epochs': 100, 'batch_size': 20})
    process.run(25, seq2seq)