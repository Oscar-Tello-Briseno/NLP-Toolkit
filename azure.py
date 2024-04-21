import requests


class AzureTextAnalyzer:
    def __init__(self, subscription_key, endpoint):
        """
        Initialize the Azure Text Analyzer with the subscription key and endpoint.
        """
        self.subscription_key = subscription_key
        self.endpoint = endpoint

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text using Azure Cognitive Services Text Analytics API.
        :param text: The text to analyze.
        :return: The sentiment analysis result (positive, negative, or neutral).
        """
        try:
            # Prepare request headers and data
            headers = {
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": self.subscription_key
            }
            data = {"documents": [{"id": "1", "language": "en", "text": text}]}

            # Make a POST request to the sentiment analysis endpoint
            response = requests.post(f"{self.endpoint}/text/analytics/v3.0/sentiment", json=data, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                sentiment = response.json()["documents"][0]["sentiment"]
                return sentiment
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error occurred during sentiment analysis: {e}")
            return None


if __name__ == "__main__":
    # Set your Azure subscription key and endpoint
    subscription_key = "your_subscription_key_here"
    endpoint = "your_endpoint_here"

    # Initialize the Azure Text Analyzer
    text_analyzer = AzureTextAnalyzer(subscription_key, endpoint)

    # Define the text to analyze
    text = "I love this product! It's amazing."

    # Analyze the sentiment of the text
    sentiment = text_analyzer.analyze_sentiment(text)

    # Display the sentiment analysis result
    if sentiment is not None:
        print("Sentiment Analysis Result:")
        print(sentiment)
