"""
hatch_build.py — Hatchling build hook.
Executes automatically during `pip install .`
Downloads and extracts weights and data from LobePrior.
"""

import os
import shutil
import zipfile
import urllib.request
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


# -------------------------------------------------------------------------
# Constantes
# -------------------------------------------------------------------------

DATA_URL = "https://github.com/MICLab-Unicamp/LobePrior/releases/download/LobePrior/data.zip"
DATA_DIR = "data"
REQUIRED_FOLDERS = ["weights", "raw_images"]


# -------------------------------------------------------------------------
# Download functions
# -------------------------------------------------------------------------

def data_already_installed() -> bool:
    """Check if the required data folders already exist."""
    return all(
        os.path.isdir(os.path.join("src", folder))
        for folder in REQUIRED_FOLDERS
    )


def download_and_extract_data() -> None:
    """Download and extract the data.zip file if the data is missing."""
    if data_already_installed():
        print("[LobePrior] Data already present. Download ignored.")
        return

    print("[LobePrior] Data not found. Starting download...")
    zip_path = "data.zip"

    print(f"[LobePrior] Downloading from  {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    print("[LobePrior] Download completed.")

    print("[LobePrior] Extracting data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    print("[LobePrior] Extraction completed.")

    os.remove(zip_path)
    print("[LobePrior] Zip file removed.")

    os.makedirs("src", exist_ok=True)

    if os.path.isdir(DATA_DIR):
        for folder in REQUIRED_FOLDERS:
            src = os.path.join(DATA_DIR, folder)
            dst = os.path.join("src", folder)

            if os.path.exists(src):
                if not os.path.exists(dst):
                    print(f"[LobePrior] Moving {src} -> {dst}")
                    shutil.move(src, dst)
                else:
                    print(f"[LobePrior] The folder {dst} already exists. Ignoring it.")
            else:
                print(f"[LobePrior] Expected folder not found: {src}")

        shutil.rmtree(DATA_DIR)
        print(f"[LobePrior] Directory {DATA_DIR} removed.")


# -------------------------------------------------------------------------
# Hook do Hatchling
# -------------------------------------------------------------------------

class CustomBuildHook(BuildHookInterface):
    """Hook executed automatically by Hatchling during build."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Called before the build — triggers the data download."""
        download_and_extract_data()
