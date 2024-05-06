from openai import OpenAI
import random


class CodeGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_source_code(self, language, specifications, max_attempts=3, max_tokens=200, seed=None):
        try:
            attempts = 0
            while attempts < max_attempts:
                # Define the input message for text generation
                input_message = [
                    {"role": "system", "content": f"You are a code generator capable of creating {language} source code based on provided specifications."},
                    {"role": "user", "content": specifications}
                ]

                # Request text completion from GPT-3.5 Turbo model
                completion = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=input_message,
                    max_tokens=max_tokens,
                    seed=seed if seed else random.randint(1, 1000)  # Use a random seed for reproducibility if not specified
                )

                # Extract and return the generated source code
                source_code = completion.choices[0].message["content"]
                if source_code:
                    return source_code

                attempts += 1

            print("Failed to generate source code after multiple attempts.")
            return None

        except Exception as e:
            print(f"An error occurred while generating source code: {e}")
            return None


def main():
    try:
        # OpenAI API key
        api_key = "your_API_key_here"

        # Language for which to generate source code (e.g., "Python", "JavaScript")
        language = input("Enter the programming language for source code generation: ")

        # Specifications for generating source code
        specifications = input("Enter the specifications for source code generation: ")

        # Maximum number of attempts for generating code
        max_attempts = int(input("Enter the maximum number of attempts for code generation (default: 3): ") or 3)

        # Maximum number of tokens for generated code
        max_tokens = int(input("Enter the maximum number of tokens for generated code (default: 200): ") or 200)

        # Custom seed value for reproducibility
        seed = input("Enter a custom seed value for reproducibility (press Enter for random seed): ")
        seed = int(seed) if seed.isdigit() else None

        # Create an instance of the code generator
        generator = CodeGenerator(api_key)

        # Generate source code based on specifications
        source_code = generator.generate_source_code(language, specifications, max_attempts, max_tokens, seed)

        if source_code:
            # Print the generated source code
            print(f"\nGenerated {language} Source Code:")
            print(source_code)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
