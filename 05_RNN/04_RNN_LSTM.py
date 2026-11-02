
import torch
import torch.nn as nn
import torch.optim as optim

# Define model parameters
vocab_size = 100
embedding_size = 16
hidden_size = 32
num_layers = 2
num_classes = 4

batch_size = 8
seq_len = 10
num_epochs = 5
learning_rate = 0.001


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        # Convert token IDs into dense vectors
        self.emb = nn.Embedding(vocab_size, embedding_size)

        # Process sequential data using LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Map hidden states to class scores
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize the hidden state and cell state
        h0 = torch.zeros(
            num_layers, x.size(0), hidden_size, device=x.device
        )
        c0 = torch.zeros(
            num_layers, x.size(0), hidden_size, device=x.device
        )

        # Convert token IDs into embeddings
        x = self.emb(x)

        # Process the sequence
        # Output shape: (batch_size, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Select the hidden state at the final time step
        out = out[:, -1, :]

        # Generate class scores
        out = self.fc(out)

        return out


# Generate sample input and labels
x_data = torch.randint(0, vocab_size, (batch_size, seq_len))
y_data = torch.randint(0, num_classes, (batch_size,))

# Initialize the model, loss function, and optimizer
model = LSTMModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()

    # Clear the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(x_data)

    # Calculate the loss
    loss = criterion(outputs, y_data)

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(x_data)
    _, predicted = outputs.max(dim=1)

print("Predicted classes:", predicted)
print("True classes:", y_data)
