import openai


class GPT3TextGenerator:
    def __init__(self, api_key):
        """
        Initialize the GPT-3 text generator with the API key.
        """
        openai.api_key = api_key

    def generate_text(self, prompt, engine="text-davinci-003", max_tokens=100):
        """
        Generate text based on the given prompt using OpenAI's GPT-3.
        :param prompt: The prompt text to generate from.
        :param engine: The GPT-3 engine to use.
        :param max_tokens: The maximum number of tokens (words) for the generated text.
        :return: The generated text.
        """
        try:
            # Generate text based on the prompt using OpenAI's GPT-3
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens
            )
            generated_text = response.choices[0].text.strip()
            return generated_text
        except Exception as e:
            print(f"Error occurred during text generation: {e}")
            return None


if __name__ == "__main__":
    # Set your OpenAI API key
    api_key = "your_api_key_here"

    # Initialize the GPT-3 text generator
    text_generator = GPT3TextGenerator(api_key)

    # Define your prompt
    prompt = "Once upon a time, in a faraway kingdom, there was a brave knight who"

    # Generate text based on the prompt
    generated_text = text_generator.generate_text(prompt)

    # Display the generated text
    if generated_text is not None:
        print("Generated Text:")
        print(generated_text)
