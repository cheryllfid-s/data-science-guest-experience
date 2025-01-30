from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

def scrape_disney_wait_times():
    # Set up Selenium WebDriver
    service = Service('chromedriver')  # Replace with the path to your ChromeDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode (no browser UI)
    driver = webdriver.Chrome(service=service, options=options)

    # URL of the page with wait times
    url = "https://touringplans.com/magic-kingdom/wait-times"

    # Open the page
    driver.get(url)

    # Wait for the page to load (adjust the sleep time as needed)
    time.sleep(10)

    # Get the page source after JavaScript has rendered the content
    page_source = driver.page_source

    # Close the browser
    driver.quit()

    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    # Find all script tags containing wait time data
    scripts = soup.find_all('script', string=re.compile(r'google\.visualization\.DataTable'))

    data = []

    for script in scripts:
        # Extract the attraction name
        attraction_match = re.search(r'title: "([^"]+)"', script.string)
        if not attraction_match:
            continue
        attraction = attraction_match.group(1)

        # Extract the data rows
        rows_match = re.search(r'addRows\(\[([^\]]+)\]\)', script.string)
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