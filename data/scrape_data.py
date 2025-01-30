import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_disney_wait_times():
    url = "https://touringplans.com/magic-kingdom/wait-times"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    wait_time_table = soup.find('table', {'class': 'wait-times'})
    rows = wait_time_table.find_all('tr')
    data = []

    for row in rows[1:]:
        cols = row.find_all('td')
        attraction = cols[0].text.strip()
        wait_time = cols[1].text.strip()
        data.append([attraction, wait_time])

    df = pd.DataFrame(data, columns=['Attraction', 'Wait Time'])
    df.to_csv('../data/disney_wait_times.csv', index=False)
    return df

if __name__ == "__main__":
    scrape_disney_wait_times()