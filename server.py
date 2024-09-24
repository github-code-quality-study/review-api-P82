import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_string = environ.get("QUERY_STRING", "")
            query_params = parse_qs(query_string)

            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            if location:
                reviews_filtered = [r for r in reviews if r['Location'] == location]
            else:
                reviews_filtered = reviews

            if start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                reviews_filtered = [r for r in reviews_filtered if datetime.strptime(r['Timestamp'], "%Y-%m-%d %H:%M:%S") >= start_date]
            if end_date:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
                reviews_filtered = [r for r in reviews_filtered if datetime.strptime(r['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date]
            
            for review in reviews_filtered:
                sentiment = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = sentiment
            reviews_filtered.sort(key=lambda r: r['sentiment']['compound'], reverse=True)

            response_body = json.dumps(reviews_filtered, indent=2).encode("utf-8")
            
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            content_length = int(environ.get('CONTENT_LENGTH', 0))
            post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
            post_params = parse_qs(post_data)
            
            location = post_params.get('Location', [None])[0]
            review_body = post_params.get('ReviewBody', [None])[0]
            
            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Bad Request: Missing location or review body"]

            if location not in [
                'Albuquerque, New Mexico',
                'Carlsbad, California',
                'Chula Vista, California',
                'Colorado Springs, Colorado',
                'Denver, Colorado',
                'El Cajon, California',
                'El Paso, Texas',
                'Escondido, California',
                'Fresno, California',
                'La Mesa, California',
                'Las Vegas, Nevada',
                'Los Angeles, California',
                'Oceanside, California',
                'Phoenix, Arizona',
                'Sacramento, California',
                'Salt Lake City, Utah',
                'San Diego, California',
                'Tucson, Arizona'
            ]:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Bad Request: Invalid location"]

            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp
            }
            
            sentiment = self.analyze_sentiment(review_body)
            new_review['sentiment'] = sentiment
            reviews.append(new_review)
            
            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
        return [b"Method Not Allowed"]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()

