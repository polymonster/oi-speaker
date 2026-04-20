"""Fetch positives and false_positives wav files from a remote http.server instance."""
import requests
import re
import os

BASE_URL = "http://192.168.1.119:8080"
DIRS = ["training/positives", "training/false_positives"]


def fetch(base_url: str = BASE_URL):
    for subdir in DIRS:
        url = f"{base_url}/{subdir}/"
        listing = requests.get(url).text
        files = re.findall(r'href="(\d+\.wav)"', listing)
        os.makedirs(subdir, exist_ok=True)
        for f in files:
            dest = os.path.join(subdir, f)
            if os.path.exists(dest):
                continue
            data = requests.get(f"{url}{f}").content
            with open(dest, "wb") as fh:
                fh.write(data)
            print(f"fetched {dest}")
        print(f"{subdir}: {len(files)} files found")


if __name__ == "__main__":
    fetch()
