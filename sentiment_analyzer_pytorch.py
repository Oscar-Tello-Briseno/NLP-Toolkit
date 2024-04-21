import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        # Initialize the LSTMModel class, inheriting from nn.Module
        super(LSTMModel, self).__init__()

        # Initialize an embedding layer
        # vocab_size: number of unique words in the vocabulary
        # embedding_dim: dimensionality of word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize an LSTM layer
        # embedding_dim: input size (dimension of input embeddings)
        # hidden_dim: size of the hidden state and cell state
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Initialize a fully connected (linear) layer
        # hidden_dim: input size (dimension of the hidden state)
        # 1: output size (1 output neuron for binary classification)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Forward pass of the model

        # Embed the input sequence x
        # x: input tensor representing a batch of sequences of token indices
        # embedded: tensor of shape (seq_len, batch_size, embedding_dim)
        embedded = self.embedding(x)

        # Pass the embedded sequences through the LSTM layer
        # lstm_out: tensor of shape (seq_len, batch_size, hidden_dim)
        # _: tuple containing the hidden state and cell state of the last time step
        lstm_out, _ = self.lstm(embedded)

        # Use only the last output of the LSTM sequence (at the last time step)
        # lstm_out[-1]: tensor representing the hidden state at the last time step
        # out: tensor representing the output of the fully connected layer
        out = self.fc(lstm_out[-1])

        # Apply sigmoid activation function to the output
        # Return the result
        return torch.sigmoid(out)


class SentimentAnalyzer:
    def __init__(self, texts=None, labels=None):
        self.labels_test = None
        self.texts_test = None
        self.texts_train = None
        self.labels_train = None
        self.texts = texts
        self.labels = labels
        self.tokenizer = None
        self.model = None

    def __repr__(self):
        return f"SentimentAnalyzer(texts={self.texts}, labels={self.labels})"

    def tokenize_texts(self):
        try:
            # Tokenize texts using NLTK word_tokenize
            # text.lower() converts each text to lowercase before tokenization
            tokenized_texts = [word_tokenize(text.lower()) for text in self.texts]

            # Count word frequencies using Counter
            # word_counts: dictionary containing word frequencies
            word_counts = Counter(word for text in tokenized_texts for word in text)

            # Create a vocabulary mapping each unique word to an index
            # self.tokenizer: dictionary mapping words to indices
            # idx + 1: indices start from 1, leaving 0 for padding
            self.tokenizer = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}

            # Add a padding token with index 0 to the tokenizer
            self.tokenizer['<PAD>'] = 0
        except Exception as e:
            # Handle exceptions during tokenization
            print(f"Error occurred during tokenization: {e}")

    def convert_texts_to_sequences(self):
        try:
            # Convert texts to sequences of token indices
            sequences = [[self.tokenizer.get(token, 0) for token in text] for text in self.texts]
            return sequences
        except Exception as e:
            print(f"Error occurred during sequence conversion: {e}")

    def split_data(self):
        try:
            # Split data into train and test sets
            self.texts_train, self.texts_test, self.labels_train, self.labels_test = \
                train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Error occurred during data splitting: {e}")

    def train_model(self, vocab_size, embedding_dim, hidden_dim, num_epochs=10, learning_rate=0.001):
        try:
            # Convert texts to sequences of token indices
            sequences_train = self.convert_texts_to_sequences()
            # Initialize LSTM model
            self.model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            # Convert sequences to PyTorch tensors
            sequences_tensor = torch.LongTensor(sequences_train)
            labels_tensor = torch.FloatTensor(self.labels)

            # Define DataLoader to handle batching and shuffling
            train_dataset = TensorDataset(sequences_tensor, labels_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            # Train the model
            for epoch in range(num_epochs):
                total_loss = 0.0
                for batch_sequences, batch_labels in train_loader:
                    # Forward pass
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs.squeeze(), batch_labels)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Print average loss for the epoch
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        except Exception as e:
            print(f"Error occurred during model training: {e}")

    def evaluate_model(self, test_data_loader, criterion):
        try:
            # Set the model to evaluation mode
            self.model.eval()

            # Initialize variables for evaluation metrics
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # Iterate over batches in the test data loader
            for inputs, labels in test_data_loader:
                # Move inputs and labels to the device (if using GPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(inputs)
                    predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions

                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    # Calculate number of correct predictions
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)

            # Calculate average loss and accuracy
            average_loss = total_loss / len(test_data_loader)
            accuracy = correct_predictions / total_predictions

            print(f"Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

            return average_loss, accuracy

        except Exception as e:
            # Handle exceptions during model evaluation
            print(f"Error occurred during model evaluation: {e}")


if __name__ == "__main__":
    texts = [
        "This movie is fantastic!",
        "I didn't like the plot twist.",
        "The acting was mediocre.",
        "Absolutely loved it!",
        "It's a waste of time."
    ]
    labels = [1, 0, 0, 1, 0]

    sentiment_analyzer = SentimentAnalyzer(texts, labels)
    sentiment_analyzer.tokenize_texts()
    sentiment_analyzer.split_data()
    sentiment_analyzer.train_model(vocab_size=10000, embedding_dim=50, hidden_dim=64, num_epochs=10,
                                   learning_rate=0.001)
