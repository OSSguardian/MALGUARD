import requests
from bs4 import BeautifulSoup
import csv
import time


import datetime
now = datetime.datetime.now()
print(now)
today = now.strftime("%d")
today = int(today)
def get_all_pypi_packages():
    # URL of the PyPI simple index
    url = "https://pypi.org/simple/"

    try:

        response = requests.get(url, timeout=60)  # Set a timeout to avoid hanging indefinitely
        response.raise_for_status()  # Raise an error for bad status codes (4xx, 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        package_links = soup.find_all('a')

        packages = [link.get('href').strip('/').replace('simple/', '') for link in package_links]

        return packages

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the package list: {e}")
        return []

def write_packages_to_csv(packages, filename=f'pypi_{today}.csv'):
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write each package name to the CSV file without a header row
            for package in packages:
                writer.writerow([package])
        print(f"Successfully written {len(packages)} packages to {filename}.")
    except IOError as e:
        print(f"An error occurred while writing to the CSV file: {e}")

if __name__ == "__main__":
    start_time = time.time()

    packages = get_all_pypi_packages()

    write_packages_to_csv(packages)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
