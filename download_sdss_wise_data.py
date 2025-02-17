"""
This script provides a function to download a file from a given URL using multiple threads.
The file is downloaded in chunks, with each thread responsible for downloading a specific chunk of the file.
The progress of the download is displayed using the tqdm library.

Example usage:
url = 'https://irfu.cea.fr/Pisp/yu-yen.chang/sw/sw_input.fits'
local_filename = 'downloaded_file.fits'
download_file(url, local_filename)

Purpose:
Download online data for modeiling the rest-frame color g-r -- SFR 
probability density function (PDF) of galaxies.  
"""

# import libraries
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

def download_chunk(url, start, end, session):
    """
    Download a chunk of a file from the given URL.

    Args:
        url (str): The URL of the file.
        start (int): The starting byte position of the chunk.
        end (int): The ending byte position of the chunk.
        session (requests.Session): The requests session object.

    Returns:
        bytes: The downloaded chunk of the file.
    """
    headers = {'Range': f'bytes={start}-{end}'}
    response = session.get(url, headers=headers, stream=True)
    response.raise_for_status()
    return response.content

def download_file(url, local_filename, num_threads=2):
    """
    Download a file from the given URL using multiple threads.

    Args:
        url (str): The URL of the file.
        local_filename (str): The name of the file to be saved locally.
        num_threads (int, optional): The number of threads to use for downloading. Defaults to 2.
    """
    with requests.Session() as session:
        response = session.head(url)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = total_size // num_threads

        with open(local_filename, 'wb') as file:
            file.truncate(total_size)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                start = i * chunk_size
                end = start + chunk_size - 1 if i < num_threads - 1 else total_size - 1
                futures.append(executor.submit(download_chunk, url, start, end, session))

            with open(local_filename, 'r+b') as file:
                for future in tqdm(futures, total=num_threads, unit='chunk'):
                    chunk = future.result()
                    file.seek(futures.index(future) * chunk_size)
                    file.write(chunk)

    print(f"Downloaded {local_filename}")


# Run the download_file function

# Define the URL and the local filename
urls = ['https://irfu.cea.fr/Pisp/yu-yen.chang/sw/sw_input.fits', 'https://irfu.cea.fr/Pisp/yu-yen.chang/sw/sw_output.fits']
local_pth = './src/galfrb/data/' 
local_filenames = ['sw_input.fits', 'sdss_wise_magphys_catalog.fits']

# Check if the file already exists

def main() :

    for url, local_filename in zip(urls, local_filenames):
        local_filename = local_pth + local_filename
        if os.path.exists(local_filename):
            print(f"File {local_filename} already exists.")
        else:
            # Run the download_file function
            print(f"Downloading {local_filename}...")
            download_file(url, local_filename)
            print(f"Downloaded {local_filename}")
            print('')
    
    return

if __name__ == '__main__':
    main()
    print('Download complete.')