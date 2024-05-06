from openai import OpenAI, APIException


class InteractiveChatbot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def initiate_dialogue(self, model="gpt-3.5-turbo", max_tokens=100, temperature=0.7):
        try:
            # Initialize dialogue
            dialogue = []
            user_input = input("You: ")
            dialogue.append({"role": "user", "content": user_input})

            while True:
                # Request completion from the specified model
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=dialogue,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=10  # Request token log probabilities for evaluating responses
                )

                # Extract the response and token log probabilities
                response = completion.choices[0].message["content"]
                token_log_probs = completion.choices[0].logprobs["token_logprobs"]

                # Print the response
                print("Bot:", response)

                # Evaluate coherence and relevance using token log probabilities
                print("Top Log Probabilities of Tokens:")
                print(token_log_probs)

                # User input for the next turn
                user_input = input("You: ")

                # Validate user input
                while not user_input.strip():
                    print("Please enter a valid message.")
                    user_input = input("You: ")

                # Check if user wants to end the conversation
                if user_input.lower() in ["end", "exit", "quit"]:
                    print("Ending conversation.")
                    break

                dialogue.append({"role": "user", "content": user_input})
                dialogue.append({"role": "assistant", "content": response})

        except APIException as e:
            print(f"API Error: {e}")

        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    # OpenAI API key
    api_key = "your_api_key_here"

    # Initialize interactive chatbot
    chatbot = InteractiveChatbot(api_key)

    try:
        # Initiate dialogue with the chatbot
        chatbot.initiate_dialogue(model="gpt-3.5-turbo", max_tokens=100, temperature=0.7)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
