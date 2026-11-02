import torch

# Hyperparameters and vocabulary
input_size = 4             # Input size / vocabulary size (one-hot dimension)
hidden_size = 4            # RNN hidden state size
num_layers = 1             # Number of stacked RNN layers
batch_size = 1             # Batch size
seq_len = 5                # Sequence length
idx2char = ['e', 'h', 'l', 'o']  # ID-to-character mapping

x_data = [1, 0, 2, 2, 3]         # Input token IDs
y_data = [3, 1, 2, 3, 2]         # Target token IDs

# Manual one-hot lookup table: rows are one-hot vectors for IDs 0..3
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

# Convert token IDs to one-hot vectors
x_one_hot = [one_hot_lookup[x] for x in x_data]

# Input shape for RNN without batch_first: (seq_len, batch_size, input_size)
# (5, 1, 4)
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)

# Label shape: (seq_len,) = (5,)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # RNN without batch_first:
        # input shape  = (seq_len, batch, input_size)
        # output shape = (seq_len, batch, hidden_size)
        self.rnn = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers
        )

    def forward(self, input):
        # Initial hidden state: (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        # out: (seq_len, batch, hidden_size) = (5, 1, 4)
        # Final hidden state is ignored here
        out, _ = self.rnn(input, hidden)

        # Reshape to (seq_len * batch_size, hidden_size) = (5, 4)
        return out.view(-1, self.hidden_size)


net = Model(input_size, hidden_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)


def train():
    for epoch in range(3):
        optimizer.zero_grad()

        # outputs: (seq_len * batch_size, hidden_size) = (5, 4)
        outputs = net(inputs)

        # CrossEntropyLoss:
        # outputs: (5, 4) logits, labels: (5,) class IDs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Predicted class ID for each time step
        _, idx = outputs.max(dim=1)
        idx = idx.detach().cpu().numpy()

        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        # Note: the loop runs 3 epochs, while the text below says "/15"
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


if __name__ == '__main__':
    print(inputs.shape, '\n', labels.shape)
    train()