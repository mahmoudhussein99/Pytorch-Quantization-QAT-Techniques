#!/usr/bin/env python3
"""Download and extract the Caltech-101 dataset.

The dataset is ~137 MB and is intentionally not tracked by git. It is placed
by default in ``experiments/caltech101/101_ObjectCategories`` which is where the
data loader in ``experiments/caltech101/utils2.py`` looks for it.

Source: CaltechDATA, record 20086 (https://data.caltech.edu/records/mzrjq-6wc02),
licensed CC BY 4.0.

Usage:
    python scripts/download_caltech101.py
    python scripts/download_caltech101.py --dest experiments/caltech101
    python scripts/download_caltech101.py --url <custom-zip-url>
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

DEFAULT_URL = (
    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join("experiments", "caltech101")
DATASET_FOLDER = "101_ObjectCategories"


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    done = min(count * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(
        "\r  {:.1f}%  ({:.1f}/{:.1f} MB)".format(pct, done / 1e6, total_size / 1e6)
    )
    sys.stdout.flush()


def download(url):
    print("Downloading Caltech-101 from {}".format(url))
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(tmp_fd)
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        print()
        return tmp_path
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _find_folder(root, name):
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    return None


def extract(zip_path, dest, folder=DATASET_FOLDER):
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, folder)
    if os.path.isdir(target):
        print("Already extracted: {}".format(target))
        return target

    print("Extracting into {}".format(dest))
    tmp_extract = tempfile.mkdtemp(prefix="caltech101_")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_extract)
        found = _find_folder(tmp_extract, folder)
        if found is None:
            raise RuntimeError(
                "Could not find '{}' inside the archive. Inspect {}".format(
                    folder, tmp_extract
                )
            )
        shutil.move(found, target)
    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)
    return target


def main():
    parser = argparse.ArgumentParser(description="Download the Caltech-101 dataset.")
    parser.add_argument(
        "--url", default=DEFAULT_URL, help="Zip URL (default: official CaltechDATA copy)"
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(REPO_ROOT, DEFAULT_DEST),
        help="Destination directory (default: experiments/caltech101)",
    )
    parser.add_argument(
        "--keep-archive", action="store_true", help="Keep the downloaded zip file"
    )
    args = parser.parse_args()

    target = os.path.join(os.path.abspath(args.dest), DATASET_FOLDER)
    if os.path.isdir(target):
        print("Caltech-101 already present at {}".format(target))
        return

    zip_path = download(args.url)
    try:
        extract(zip_path, args.dest)
    finally:
        if not args.keep_archive and os.path.exists(zip_path):
            os.remove(zip_path)

    print("Done. Dataset available at {}".format(target))


if __name__ == "__main__":
    main()
