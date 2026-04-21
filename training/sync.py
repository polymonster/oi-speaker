"""Fetch training wav files from a remote device running http.server."""
import argparse
import os
import re

import requests

DIRS = ["training/training_data/recordings/positives", "training/training_data/recordings/false_positives"]


def fetch(ip: str):
    base_url = f"http://{ip}:8080"
    for subdir in DIRS:
        url = f"{base_url}/{subdir}/"
        listing = requests.get(url).text
        files = re.findall(r'href="(\d+(?:_\d+)?\.wav)"', listing)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ip", help="IP address of the remote device")
    args = parser.parse_args()
    fetch(args.ip)
