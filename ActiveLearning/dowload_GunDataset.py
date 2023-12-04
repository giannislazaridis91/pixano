""" Credits goes to https://stackoverflow.com/users/1475331/turdus-merula for providing an answer in this post https://stackoverflow.com/a/39225272/15842840"""

import os
import shutil
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://drive.usercontent.google.com/download?id=1hWqABdPezJ8HBTvbj_I9fy2bDii4-lWE&export=download&authuser=0&confirm=t&uuid=bf6fef23-c22a-4ed9-9842-d6f11bef2b52&at=APZUnTW6wkaLYW5S9v7dGb1TUQLZ:1701268449279"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def getGunDataset(destination):
    
    file_id = "1hWqABdPezJ8HBTvbj_I9fy2bDii4-lWE"

    # print(f"download {file_id} to {destination}")
    download_file_from_google_drive(file_id, destination)
    
    extraction_path = os.path.dirname(destination)

    shutil.unpack_archive(destination, extraction_path)

    os.remove(destination)

    return os.path.join(extraction_path,"gun")