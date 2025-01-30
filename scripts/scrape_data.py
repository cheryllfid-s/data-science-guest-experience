import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

def scrape_disney_wait_times():
    # URL of the page with wait times
    url = "https://touringplans.com/magic-kingdom/wait-times"

    # Send a GET request to the website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all script tags containing wait time data
    scripts = soup.find_all('script', text=re.compile(r'google\.visualization\.DataTable'))

    data = []

    for script in scripts:
        # Extract the attraction name
        attraction_match = re.search(r'title: "([^"]+)"', script.text)
        if not attraction_match:
            continue
        attraction = attraction_match.group(1)

        # Extract the data rows
        rows_match = re.search(r'addRows\(\[([^\]]+)\]\)', script.text)
        if not rows_match:
            continue
        rows = rows_match.group(1)

        # Parse the rows into a list of dictionaries
        for row in rows.split('],['):
            row = row.replace('[', '').replace(']', '')
            time, wait_time = row.split(',')
            time = time.strip().strip('"')
            wait_time = int(wait_time.strip())

            data.append({
                'Attraction': attraction,
                'Time': time,
                'Wait Time (Minutes)': wait_time
            })

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    df.to_csv('disney_wait_times.csv', index=False)
    return df

if __name__ == "__main__":
    df = scrape_disney_wait_times()
    print(df.head())