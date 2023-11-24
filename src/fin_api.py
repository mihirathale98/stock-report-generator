import json
import finnhub


with open('key.json', 'r') as f:
    API_KEY = json.load(f)['api_key']

finnhub_client = finnhub.Client(api_key=API_KEY)

print(finnhub_client.transcripts_list('AAPL'))
