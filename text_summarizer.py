from openai import OpenAI, APIException


class TextSummarizer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def summarize_text(self, input_text, model="gpt-3.5-turbo", max_length=200, mode="sentences",
                       temperature=0.7, frequency_penalty=0.2, presence_penalty=0.2):
        try:
            # Define the input message for text summarization
            message_input = [
                {"role": "system", "content": "You are an AI assistant capable of summarizing text with precision."},
                {"role": "user", "content": input_text}
            ]

            # Determine max_tokens based on mode
            if mode == "sentences":
                max_tokens = self._calculate_max_tokens(input_text, max_length)
            elif mode == "words":
                max_tokens = max_length
            else:
                raise ValueError("Invalid mode. Please choose either 'sentences' or 'words'.")

            # Request text completion from the specified model
            completion = self.client.chat.completions.create(
                model=model,
                messages=message_input,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=10  # Request token log probabilities for better summarization
            )

            # Extract the generated summary and token log probabilities
            summary = completion.choices[0].message["content"]
            token_log_probs = completion.choices[0].logprobs["token_logprobs"]

            return summary, token_log_probs

        except APIException as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")

    def _calculate_max_tokens(self, text, max_length):
        # Calculate the approximate number of tokens required to reach the desired summary length in sentences
        tokens_per_sentence = 15  # Approximate average number of tokens per English sentence
        return min(len(text.split()) + tokens_per_sentence * max_length, 4096)  # Max tokens limit


def main():
    # OpenAI API key
    api_key = "your_api_key_here"

    # Input text to be summarized
    input_text = "Long text to be summarized..."

    # Initialize text summarizer
    summarizer = TextSummarizer(api_key)

    try:
        # Summarize the input text
        summary, token_log_probs = summarizer.summarize_text(input_text, model="gpt-3.5-turbo", max_length=5,
                                                             mode="sentences", temperature=0.7,
                                                             frequency_penalty=0.2, presence_penalty=0.2)

        # Print the generated summary
        print("Text Summary:")
        print(summary)

        # Print the top log probabilities of tokens for each position in the generated summary
        print("Top Log Probabilities of Tokens:")
        print(token_log_probs)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
