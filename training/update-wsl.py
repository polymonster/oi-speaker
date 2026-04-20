"""Copy new training wavs to WSL openWakeWord training directories."""
import shutil
import os

MAPPINGS = [
    ("training/positives",      r"\\wsl$\Ubuntu\home\alex\oww-training\positive_samples"),
    ("training/false_positives", r"\\wsl$\Ubuntu\home\alex\oww-training\negative_samples"),
]


def update():
    for src_dir, dst_dir in MAPPINGS:
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
    update()
