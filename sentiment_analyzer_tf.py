import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentimentAnalyzer(object):
    def __init__(self, texts: list, labels: list) -> None:
        # Initialize with texts and corresponding labels
        self.texts = texts
        self.labels = labels
        self.tokenizer = Tokenizer()  # Initialize Tokenizer for text preprocessing
        self.model = None  # Model placeholder

    def __repr__(self) -> str:
        return f"SentimentAnalyzer(texts={self.texts}, labels={self.labels})"

    def preprocess_texts(self) -> None:
        try:
            # Tokenize the texts and build vocabulary
            # Tokenization is the process of breaking down a text into smaller units, usually words or subwords,
            # which are called tokens.
            # Tokenization is crucial because it converts raw text data into a format that can be understood and
            # processed by machine learning models.
            self.tokenizer.fit_on_texts(self.texts)
        except Exception as e:
            print(f"Error occurred during tokenization: {e}")

    def tokenize_and_pad(self) -> tuple:
        """
        We tokenize the texts and pad the sequences to ensure uniform input length for the model.
        """
        try:
            # Convert texts to sequences and pad them to ensure uniform length

            # Convert texts to sequences: The first step is to convert texts into sequences of integers,
            # where each word in the text is assigned a unique number. This is done using the texts_to_sequences
            # method of the Tokenizer object. Converting texts to numerical sequences is necessary so that the
            # machine learning model can understand the input data in numerical form instead of raw text.
            sequences = self.tokenizer.texts_to_sequences(self.texts)

            # Determine the maximum length: After converting the texts into sequences, we need to determine the maximum
            # length of the sequences. This is important for the padding step, where we add zeros to the end of the
            # shorter sequences so that they all have the same length. The maximum length is calculated by finding the
            # maximum length among all converted sequences.
            max_length = max([len(seq) for seq in sequences])

            # Sequence padding: Once we know the maximum length, we use the pad_sequences function to add zeros to the
            # end of the shortest sequences, ensuring that all sequences have the same length. This is crucial because
            # machine learning models require inputs of uniform size. The maxlen parameter is set to the maximum length
            # calculated above, and the padding parameter is set to 'post', meaning that zeroes are added to the end of
            # the sequences.
            padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

            return padded_sequences, max_length
        except Exception as e:
            print(f"Error occurred during tokenization and padding: {e}")
            return None, None

    def build_model(self, max_length):
        """
        We build a Sequential model with an Embedding layer, an LSTM layer, and a Dense layer for classification.
        :param max_length:
        :return:
        """
        try:
            # Build LSTM model for sentiment analysis

            # Embedding Layer: The Embedding layer is used to convert integer-encoded words into dense vectors of fixed
            # size. It creates a mapping from each integer to a dense vector representation. In this case, vocab_size
            # represents the size of the vocabulary, output_dim specifies the dimensionality of the dense embedding,
            # and input_length is the length of input sequences.
            vocab_size = len(self.tokenizer.word_index) + 1

            # LSTM Layer: LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture that
            # is capable of learning long-term dependencies in sequential data. It's well-suited for tasks like
            # sentiment analysis where the order of words matters. Here, units=64 specifies the dimensionality of the
            # output space.
            self.model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length),
                LSTM(units=64),
                # Dense Layer: The Dense layer is a fully connected layer that performs classification based on the
                # output of the LSTM layer. It has a single neuron with a sigmoid activation function, which outputs a
                # value between 0 and 1, representing the probability of the input belonging to the positive class.
                Dense(1, activation='sigmoid')
            ])

            # Model Compilation: After defining the architecture, the model is compiled using the Adam optimizer and
            # binary cross-entropy loss function. The Adam optimizer is an adaptive learning rate optimization
            # algorithm, and binary cross-entropy is commonly used for binary classification tasks like sentiment
            # analysis. Additionally, we specify accuracy as a metric to monitor during training.
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except Exception as e:
            print(f"Error occurred during model building: {e}")

    def train_model(self, padded_sequences) -> None:
        """
        We train the model on the tokenized and padded sequences along with their labels.
        :param padded_sequences:
        :return:
        """
        try:
            # Train the model
            self.model.fit(padded_sequences, self.labels, epochs=10)
        except Exception as e:
            print(f"Error occurred during model training: {e}")

    def predict_sentiment(self, test_text) -> None:
        try:
            if self.model is None:
                print("Model not trained. Please train the model first.")
                return
            # Preprocess the test text and make prediction
            test_sequence = self.tokenizer.texts_to_sequences([test_text])
            padded_test_sequence, _ = self.tokenize_and_pad()
            prediction = self.model.predict(padded_test_sequence)
            print(f"Predicted sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")
        except Exception as e:
            print(f"Error occurred during sentiment prediction: {e}")

    def save_model(self, filepath) -> None:
        try:
            self.model.save(filepath)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error occurred during model saving: {e}")

    @classmethod
    def load_model(cls, filepath):
        try:
            model = tf.keras.models.load_model(filepath)
            sentiment_analyzer = cls()
            sentiment_analyzer.model = model
            print("Model loaded successfully.")
            return sentiment_analyzer
        except Exception as e:
            print(f"Error occurred during model loading: {e}")
            return None


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
    sentiment_analyzer.preprocess_texts()
    padded_sequences, max_length = sentiment_analyzer.tokenize_and_pad()
    if padded_sequences is not None and max_length is not None:
        sentiment_analyzer.build_model(max_length)
        sentiment_analyzer.train_model(padded_sequences)

        test_text = "It was surprisingly good!"
        sentiment_analyzer.predict_sentiment(test_text)

        # Example saving and loading
        sentiment_analyzer.save_model("sentiment_model.h5")
        loaded_analyzer = SentimentAnalyzer.load_model("sentiment_model.h5")
        if loaded_analyzer:
            loaded_analyzer.predict_sentiment(test_text)
