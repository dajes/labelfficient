import os
import pickle
from typing import Optional

import requests
import torch
import torch.hub

from commons.utils import get_all_files
from commons.utils import string_hash


def download_file_from_google_drive(google_id: str, destination: str) -> None:
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': google_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': google_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response: requests.Response, destination: str) -> None:
    chunk_size = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_state_dict(state_dict_url, return_path=False):
    model_dir = os.path.join('.cache', string_hash(state_dict_url))
    state_dict_path = None
    if os.path.exists(model_dir):
        files = get_all_files(model_dir)
        if len(files) == 1:
            state_dict_path = files[0]
    if state_dict_path is None:
        try:
            torch.hub.load_state_dict_from_url(state_dict_url, model_dir=model_dir, map_location='cpu')
            state_dict_path = get_all_files(model_dir)[0]
        except pickle.UnpicklingError:
            state_dict_path = os.path.join(model_dir, 'uc')
            file_id = state_dict_url.split('id=')[-1].split('&')[0]
            download_file_from_google_drive(file_id, state_dict_path)
    if not return_path:
        return torch.load(state_dict_path, map_location='cpu')

    return state_dict_path


def get_local_path_of_url(path_or_url):
    if path_or_url.startswith('http'):
        state_dict_path = download_state_dict(path_or_url, return_path=True)
    else:
        state_dict_path = path_or_url
    return state_dict_path


def load_file(path_or_url: str, map_location: str = 'cpu') -> dict:
    state_dict = torch.load(get_local_path_of_url(path_or_url), map_location=map_location)
    return state_dict
