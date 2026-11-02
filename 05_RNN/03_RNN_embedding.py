import torch

# Hyperparameters and vocabulary
num_class = 4              # Number of output classes
input_size = 4             # Vocabulary size / number of token IDs
hidden_size = 8            # RNN hidden state size
embedding_size = 10        # Embedding dimension
num_layers = 2             # Number of stacked RNN layers
batch_size = 1             # Batch size
seq_len = 5                # Sequence length
idx2char = ['e', 'h', 'l', 'o']  # ID-to-character mapping

x_data = [[1, 0, 2, 2, 3]]       # Input token IDs, shape: (batch_size, seq_len)
y_data = [3, 1, 2, 3, 2]         # Target token IDs, shape: (batch_size * seq_len,)

inputs = torch.LongTensor(x_data)   # Shape: (batch_size, seq_len) = (1, 5)
labels = torch.LongTensor(y_data)   # Shape: (batch_size * seq_len,) = (5,)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Embedding table: shape (num_embeddings, embedding_dim) = (4, 10)
        self.emb = torch.nn.Embedding(input_size, embedding_size)

        # RNN: input size = embedding_size, output hidden size = hidden_size
        # batch_first=True means input/output shape is (batch, seq, feature)
        self.rnn = torch.nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected layer: hidden_size -> num_class
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # Initial hidden state: (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)

        # Input IDs -> embeddings: (batch, seq_len, embedding_size)
        x = self.emb(x) # (batch, seq_len, embedding_size) = (1, 5, 10)

        # RNN output: (batch, seq_len, hidden_size)
        # Final hidden state is ignored here
        x, _ = self.rnn(x, hidden)  # (batch, seq_len, hidden_size)   = (1, 5, 8)

        # Class logits: (batch, seq_len, num_class)
        x = self.fc(x)  # (batch, seq_len, num_class)        = (1, 5, 4)

        # Flatten for CrossEntropyLoss: (batch * seq_len, num_class)
        return x.view(-1, num_class)


net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)


def train():
    for epoch in range(5):
        optimizer.zero_grad()

        # Output shape: (batch * seq_len, num_class) = (5, 4)
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Get predicted class ID for each time step
        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()

        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        # Note: the loop runs 5 epochs, while the text below says "/15"
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


if __name__ == '__main__':
    print(inputs.shape, '\n', labels.shape)
    train()