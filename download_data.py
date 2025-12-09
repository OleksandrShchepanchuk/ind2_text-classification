import os
import json
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the same directory as the script
env_path = Path(__file__).parent / ".env"
load_result = load_dotenv(dotenv_path=env_path, override=True)
print(f"load_dotenv returned: {load_result}")
from  kaggle import api


COMPETITION = "lnu-deep-learn-2-text-classification-2025"
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download():
    api.competition_download_files(COMPETITION, path=str(OUT_DIR), quiet=False)

    for z in OUT_DIR.glob("*.zip"):
        import zipfile
        with zipfile.ZipFile(z) as zipf:
            zipf.extractall(OUT_DIR)
        z.unlink()

if __name__ == "__main__":
    download()
    print("Download complete:", OUT_DIR)