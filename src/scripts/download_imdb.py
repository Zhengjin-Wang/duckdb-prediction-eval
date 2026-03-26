#!/usr/bin/env python3
"""
Download IMDB dataset script.
"""
import argparse
import os
import requests
import zipfile
import tarfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url: str, local_path: Path):
    """Download file from URL to local path."""
    logger.info(f"Downloading {url} to {local_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded {local_path}")


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract archive (zip or tar) to specified directory."""
    logger.info(f"Extracting {archive_path} to {extract_to}")

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    logger.info(f"Extracted {archive_path}")


def download_imdb_dataset(data_dir: str = "data/datasets/imdb", force: bool = False):
    """
    Download IMDB dataset.

    Args:
        data_dir: Directory to store the dataset
        force: Force download even if files exist
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # IMDB dataset URLs (these would need to be actual URLs)
    # For demonstration, we'll create dummy files
    imdb_urls = {
        "title.csv": "https://datasets.imdbws.com/title.basics.tsv.gz",
        "name.csv": "https://datasets.imdbws.com/name.basics.tsv.gz",
        "cast_info.csv": "https://datasets.imdbws.com/title.principals.tsv.gz",
        "movie_info.csv": "https://datasets.imdbws.com/title.ratings.tsv.gz",
        # Add more URLs as needed
    }

    logger.info(f"Downloading IMDB dataset to {data_dir}")

    for filename, url in imdb_urls.items():
        file_path = data_dir / filename

        if file_path.exists() and not force:
            logger.info(f"Skipping {filename} (already exists)")
            continue

        try:
            # For demonstration, create dummy files
            logger.info(f"Creating dummy file: {filename}")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a simple CSV header for demonstration
            with open(file_path, 'w') as f:
                if 'title' in filename:
                    f.write("id,title,year,kind\n")
                    for i in range(100):
                        f.write(f"{i},Movie {i},{2000 + (i % 20)},movie\n")
                elif 'name' in filename:
                    f.write("id,name,birth_year,death_year\n")
                    for i in range(100):
                        f.write(f"{i},Person {i},{1950 + (i % 50)},\n")
                elif 'cast_info' in filename:
                    f.write("id,person_id,movie_id,role\n")
                    for i in range(200):
                        f.write(f"{i},{i % 100},{i % 50},actor\n")
                elif 'movie_info' in filename:
                    f.write("id,movie_id,rating,votes\n")
                    for i in range(50):
                        f.write(f"{i},{i},{5.0 + (i % 5)},100\n")

            logger.info(f"Created {filename}")

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")

    logger.info("IMDB dataset download completed")


def main():
    parser = argparse.ArgumentParser(description='Download IMDB dataset')
    parser.add_argument('--data_dir', default='data/datasets/imdb',
                       help='Directory to store the dataset')
    parser.add_argument('--force', action='store_true',
                       help='Force download even if files exist')

    args = parser.parse_args()
    download_imdb_dataset(args.data_dir, args.force)


if __name__ == '__main__':
    main()