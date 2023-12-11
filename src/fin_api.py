import requests

"""
Util file to fetch transcripts from the API and save them locally

"""

# Get transcript from API
def get_jsonparsed_data(company, quarter, year):
    url = (f"https://discountingcashflows.com/api/transcript/{company}/{quarter}/{year}")
    response = requests.get(url)
    data = response.json()
    transcript = data[0]['content']
    return transcript

# List the company tickers
# Can add more publicly traded companies to this list
company_tickers = ["AMZN"]

# Fetch and save transcripts for each company
for ticker in company_tickers:
    # Fetch transcripts for each year from 2015 to 2022
    for year in range(2015, 2023):
        # Fetch transcripts for each quarter
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            transcript = get_jsonparsed_data(ticker, quarter, year)
            lines = transcript.split("\n")
            with open(f"../earnings_call_transcripts/{ticker}_{quarter}_{year}.txt", 'w') as f:
                f.writelines(lines)
