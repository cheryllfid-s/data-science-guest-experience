import matplotlib.pyplot as plt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime

def parse_js_date(js_date_str):
    """Parses a JavaScript Date string like 'new Date(2025, 0, 30, 07, 30, 00)'"""
    match = re.match(r'new Date\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\)', js_date_str)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month + 1, day, hour, minute, second)  # Months are 0-indexed in JS
    return None

def scrape_disney_wait_times():
    # Set up Selenium WebDriver
    options = Options()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')  # Prevents some rendering issues

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        return pd.DataFrame()

    url = "https://touringplans.com/magic-kingdom/wait-times"
    driver.get(url)

    # Wait until the wait times are loaded
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#content")))
    except Exception as e:
        print(f"Timeout waiting for page to load: {e}")
        driver.quit()
        return pd.DataFrame()

    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')

    # Debug: Save the page for inspection
    with open("debug.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print("Saved debug.html for inspection")

    # Find the script containing the dateData variable
    script_tag = soup.find('script', string=re.compile(r'var dateData = new google.visualization.DataTable\(\);'))

    if not script_tag:
        print("No matching script tag found. The structure may have changed.")
        return pd.DataFrame()

    # Extract the JavaScript code where data is populated
    data_string = script_tag.string
    rows_match = re.findall(r'addRows\(\[([^\]]+)\]\);', data_string)

    if not rows_match:
        print("No rows found in the script.")
        return pd.DataFrame()

    data = []
    for rows in rows_match:
        rows = rows.replace('[', '').replace(']', '')  # Clean up the rows
        for row in rows.split('],['):
            row = row.strip()
            try:
                # Extract date and wait time
                date_str, wait_time = row.split(',')
                date = parse_js_date(date_str.strip())
                wait_time = int(wait_time.strip())
                if date:
                    data.append({
                        'Time': date,
                        'Wait Time (Minutes)': wait_time
                    })
            except ValueError:
                print(f"Skipping row due to format issue: {row}")

    df = pd.DataFrame(data)

    if df.empty:
        print("No data extracted. The page structure may have changed.")
    else:
        df['Hour'] = df['Time'].dt.hour
        df['Minute'] = df['Time'].dt.minute
        df['Formatted Time'] = df['Time'].dt.strftime('%I:%M %p')

        # Filter data for specific hours (10:00am, 1:00pm, 4:00pm)
        specific_times = ['10:00 AM', '1:00 PM', '4:00 PM']
        filtered_df = df[df['Formatted Time'].isin(specific_times)]

        # Pivot the data so that each time (10:00am, 1:00pm, 4:00pm) becomes a column
        pivot_df = filtered_df.pivot_table(index=['Time'], columns=['Formatted Time'], values='Wait Time (Minutes)', aggfunc='first')

        # Reformat to a readable format
        pivot_df.columns = pivot_df.columns.str.title()  # Capitalize time format

        # Add the attraction name as a row (assuming you want it as part of the result)
        pivot_df.insert(0, 'Attraction', 'Seven Dwarfs Mine Train')  # Replace with actual attraction name if needed

        # Save to CSV
        pivot_df.to_csv('disney_wait_times_forecast.csv', index=False)

        # Plot the data using Matplotlib
        plot_wait_times(pivot_df)

    return pivot_df

def plot_wait_times(df):
    """Plots the wait times for each attraction at specific times."""
    plt.figure(figsize=(10, 6))

    # Plot each attraction's wait time at specific times
    plt.plot(df.columns[1:], df.iloc[0, 1:], marker='o', label=df.iloc[0, 0])  # Assuming one attraction for now

    # Customize the plot
    plt.title("Wait Times Forecast for January 30, 2025")
    plt.xlabel('Time')
    plt.ylabel('Wait Time (Minutes)')
    plt.xticks(rotation=45)
    plt.legend(title='Attraction', loc='upper left')
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    df = scrape_disney_wait_times()
    if not df.empty:
        print(df.head())
    else:
        print("No data found. Please check debug.html for further details.")
