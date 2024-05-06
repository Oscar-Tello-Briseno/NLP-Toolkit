from openai import OpenAI
from langdetect import detect_langs, DetectorFactory


class Translator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def translate_text(self, source_text, target_language, source_language=None,
                       model="gpt-4-turbo-preview", frequency_penalty=0.5,
                       presence_penalty=0.5, seed=None, max_tokens=100,
                       stop_sequences=None):
        # Detect the source language if not provided
        if source_language is None:
            try:
                detected_lang = detect_langs(source_text)[0].lang
            except Exception as e:
                print(f"Error detecting source language: {e}")
                return None
            source_language = detected_lang

        # Define the input message for text translation
        input_message = [
            {"role": "system",
             "content": f"You are an efficient language translator from {source_language} to {target_language}."},
            {"role": "user", "content": source_text}
        ]

        # Request text completion from the model
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=input_message,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                max_tokens=max_tokens,
                stop=stop_sequences
            )

            # Extract and return the translated text from the response
            translated_text = completion.choices[0].message.content
            return translated_text

        except Exception as e:
            print(f"Error occurred during translation: {e}")
            return None

    def batch_translate(self, texts, target_language, source_language=None, model="gpt-4-turbo-preview",
                        frequency_penalty=0.5, presence_penalty=0.5, seed=None, max_tokens=100):
        translations = []

        for text in texts:
            if source_language is None:
                translated_text = self.translate_text(text, target_language, model,
                                                      frequency_penalty, presence_penalty, seed, max_tokens)
            else:
                translated_text = self.translate_text(text, target_language, model,
                                                      frequency_penalty, presence_penalty, seed, max_tokens)

            translations.append(translated_text)

        return translations

    def bidirectional_translate(self, text, source_language, target_language,
                                model="gpt-4-turbo-preview", frequency_penalty=0.5,
                                presence_penalty=0.5, seed=None, max_tokens=100,
                                stop_sequences=None):
        # Translate text from source to target language
        translated_to_target = self.translate_text(text, target_language,
                                                   source_language, model,
                                                   frequency_penalty, presence_penalty,
                                                   seed, max_tokens, stop_sequences)

        # Translate the translated text back to the source language
        translated_back_to_source = self.translate_text(translated_to_target, source_language,
                                                        target_language, model,
                                                        frequency_penalty, presence_penalty,
                                                        seed, max_tokens, stop_sequences)

        return translated_to_target, translated_back_to_source


# Example usage
if __name__ == "__main__":
    api_key = "your_api_key_here"
    translator = Translator(api_key)

    source_texts = ["Translate this text into Spanish.", "Translate this one too."]
    target_language = "Spanish"

    translated_texts = translator.batch_translate(source_texts, target_language)
    for idx, text in enumerate(translated_texts):
        print(f"Translated Text {idx + 1}:", text)

    source_text = "Translate this text from English to Spanish."
    source_language = "English"
    target_language = "Spanish"

    translated_to_target, translated_back_to_source = translator.bidirectional_translate(source_text,
                                                                                         source_language,
                                                                                         target_language)

    print("Translated to Target Language:", translated_to_target)
    print("Translated Back to Source Language:", translated_back_to_source)
