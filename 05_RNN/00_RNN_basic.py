
import torch

def rnn_cell():
    batch_size = 2
    seq_len = 3
    input_size = 4
    hidden_size = 2

    cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
    # (seq, batch, features)
    dataset = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(batch_size, hidden_size)

    for idx, input in enumerate(dataset):
        print('=' * 20, idx, '=' * 20)
        print('Input size: ', input.shape)
        print(input)
        hidden = cell(input, hidden)
        print('outputs size: ', hidden.shape)
        print(hidden)

###########################################################################################

def rnn():
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2
    num_layers = 1

    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    # (seqLen, batchSize, inputSize)
    inputs = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    out, hidden = cell(inputs, hidden)

    print('Output size:', out.shape)
    print('Output:', out)
    print('Hidden size: ', hidden.shape)
    print('Hidden: ', hidden)

###########################################################################################

if __name__ == '__main__':
    rnn_cell()
    print("*"*60)
    rnn()