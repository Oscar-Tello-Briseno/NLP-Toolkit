from openai import OpenAI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk


class QAAssistant:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"  # Default model
        self.temperature = 0.7  # Default temperature

    def set_model(self, model):
        self.model = model

    def set_temperature(self, temperature):
        self.temperature = temperature

    def answer_question(self, question, examples=None):
        try:
            # Define the input message for question answering
            message_input = [
                {"role": "system", "content": "You're an intelligent assistant ready to answer questions on various topics."},
                {"role": "user", "content": f"What is {question}?"}
            ]

            # Add examples if provided
            if examples:
                message_input.extend(examples)

            # Request completion from the model with token log probabilities and specified temperature
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=message_input,
                logprobs=10,  # Retrieve token log probabilities for evaluation
                temperature=self.temperature  # Adjust temperature for creativity
            )

            # Extract the answer from the model's response
            answer = completion.choices[0].message["content"]

            # Retrieve token log probabilities for confidence evaluation
            logprobs = completion.choices[0].logprobs

            return answer, logprobs

        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None

    def initialize_nltk_resources(self):
        # Download NLTK resources
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        except Exception as e:
            print(f"Error downloading NLTK resources: {e}")

    @staticmethod
    def validate_answer(answer, threshold=0.5):
        try:
            # Tokenize the answer
            tokens = word_tokenize(answer)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

            # Lemmatize the tokens
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

            # Calculate the number of valid tokens
            valid_tokens = sum(1 for token in lemmatized_tokens if wordnet.synsets(token))

            # Calculate the ratio of valid tokens to total tokens
            validity_ratio = valid_tokens / len(lemmatized_tokens)

            # Compare the validity ratio with the threshold
            if validity_ratio >= threshold:
                return True
            else:
                return False
        except Exception as e:
            print(f"Error validating answer: {e}")
            return False

    @staticmethod
    def integrate_with_other_systems(answer):
        # Placeholder function for integrating with other systems
        # Implement your integration logic here
        # For example, you could store the answer in a database, send it via email, or display it in a web application
        print("Integration with other systems:")
        print("Answer:", answer)
        # Sample integration: Store answer in a database
        # database.save(answer)


def main():
    try:
        # API Key for OpenAI
        api_key = "your_api_key_here"

        # Question to ask the assistant
        question = "the capital of France"

        # Examples to guide the assistant's responses
        examples = [
            {"role": "assistant", "content": "Paris is the capital of France."},
            {"role": "user", "content": "Who is the current president of France?"}
        ]

        # Instantiate the QA assistant
        qa_assistant = QAAssistant(api_key)

        # Set the model and temperature (optional)
        qa_assistant.set_model("gpt-3.5-turbo-preview")
        qa_assistant.set_temperature(0.5)

        # Ask a question to the assistant
        answer, logprobs = qa_assistant.answer_question(question, examples)

        if answer is not None:
            # Print the answer provided by the assistant
            print("Answer:")
            print(answer)

            # Validate the answer
            is_valid = qa_assistant.validate_answer(answer)
            print("Answer Validity:", "Valid" if is_valid else "Invalid")

            # Integrate with other systems
            qa_assistant.integrate_with_other_systems(answer)

            # Print token log probabilities for confidence evaluation
            print("Token Log Probabilities:")
            print(logprobs)
        else:
            print("Failed to retrieve an answer. Please try again later.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
