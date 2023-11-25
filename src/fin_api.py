import requests
import json


def get_jsonparsed_data(url, company, quarter, year):
    url = (f"https://discountingcashflows.com/api/transcript/{company}/{quarter}/{year}")
    response = requests.get(url)
    data = response.json()
    transcript = data[0]['content']
    return transcript