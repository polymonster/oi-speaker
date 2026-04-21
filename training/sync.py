"""Sync training wav files — fetch from remote or push to WSL."""
import argparse
import os
import re
import shutil

import requests

BASE_URL = "http://192.168.1.119:8080"
DIRS = ["training/positives", "training/false_positives"]
WSL_MAPPINGS = [
    ("training/positives",      r"\\wsl$\Ubuntu\home\alex\oww-training\positive_samples"),
    ("training/false_positives", r"\\wsl$\Ubuntu\home\alex\oww-training\negative_samples"),
]


def fetch(ip: str):
    base_url = f"http://{ip}:8080"
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


def update_wsl():
    for src_dir, dst_dir in WSL_MAPPINGS:
        if not os.path.isdir(src_dir):
            print(f"skipping {src_dir} (not found)")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        existing = set(os.listdir(dst_dir))
        copied = 0
        for f in os.listdir(src_dir):
            if f.endswith(".wav") and f not in existing:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
                copied += 1
        print(f"{src_dir} → {dst_dir}: {copied} new files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="fetch wavs from remote http.server")
    p_fetch.add_argument("ip", nargs="?", default=BASE_URL.split("//")[1].split(":")[0],
                         help="IP address of the remote (default: %(default)s)")

    sub.add_parser("update-wsl", help="copy new wavs into WSL training directories")

    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.ip)
    else:
        update_wsl()
