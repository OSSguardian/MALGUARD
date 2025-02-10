import requests
import os
import json
from datetime import datetime
import csv
import concurrent.futures
from tqdm import tqdm

count_update = 0
import datetime
now = datetime.datetime.now()
print(now)
today = now.strftime("%d")
today = int(today)

csv_file_name = fr"pypi_{today}.csv"
download_path = r"weekly_update\new"

def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content():
            file.write(chunk)
        # print('downloads successful')
    file.close()


def process_row(row):
    global count_update

    package_name = row[0]
    try:
        response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
        # response = requests.get(f'https://pypi.org/pypi/pdflibrary/json')
        data = response.json()

        for version, releases in data['releases'].items():
            for release in releases:
                upload_time_str = release['upload_time']
                upload_time = datetime.strptime(upload_time_str, "%Y-%m-%dT%H:%M:%S")
                if upload_time >= datetime(2025, 1, 15):
                    url = release['url']
                    file_size = release['size']
                    file_name = release['filename']
                    if file_name.endswith(".tar.gz"):

                        print(package_name)
                        print(file_name)
                        print(url)
                        print(file_size/1024/1024, "MB")
                        download_file(url, os.path.join(download_path, file_name))

                        if file_size/1024/1024 < 1 and file_name not in os.listdir(download_path):
                            download_file(url, os.path.join(download_path, file_name))
                            count_update += 1
                        # print("**********************")

    except Exception as e:
        pass
        # print(f"get {package_name} update information wrong since {e}"


with open(csv_file_name, "r") as file:
    reader = csv.reader(file)
    rows = list(reader)
    rows = rows[:]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_row, rows), total=len(rows), desc="Processing packages"))


print(f"Total number of packages updated: {count_update}")

