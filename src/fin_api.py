import requests
import json


def get_jsonparsed_data(company, quarter, year):
    url = (f"https://discountingcashflows.com/api/transcript/{company}/{quarter}/{year}")
    response = requests.get(url)
    data = response.json()
    transcript = data[0]['content']
    return transcript



all_transcripts = []
for quarter in ["Q1", "Q2", "Q3", "Q4"]:
        transcript = get_jsonparsed_data("AMZN", quarter, 2022)
        all_transcripts.append(transcript)

with open('all_transcripts.json', 'w') as f:
    json.dump(all_transcripts, f)