import os
from hashlib import sha512
from typing import Union, List


def _append_file(images, filename, endswith, startswith, dirname, only_names, remove_root_folder, path):
    _filename = filename.lower()
    if _filename.endswith(endswith) and _filename.startswith(startswith):
        if remove_root_folder:
            dirname = dirname[len(path) + 1:]
        if only_names:
            images.append(filename)
        else:
            images.append(os.path.join(dirname, filename))


def get_all_files(path: str, endswith: str = '', startswith: str = '', only_names: bool = False,
                  recurse: bool = True, remove_root_folder: bool = False) -> List[str]:
    """
    Returns all files in the directory

    Args:
        path: path to the directory
        endswith: returns only files which names end with this substring
        startswith: returns only files which names start with this substring
        only_names: whether to return only file name or the whole path to the file
        recurse: whether to recursively visit all of the subdirectories or return files only from the root one
        remove_root_folder: whether to remove the part of paths that is the same for all folders (and equal to `path`)

    Returns:
        List[str]: all paths to all of the files in the directory

    """
    assert not (only_names and remove_root_folder), 'Only one of these arguments can be True'
    if path.endswith('/'):
        path = path[:-1]
    images = []
    for i, (dirname, _, filenames) in enumerate(os.walk(path)):
        for j, filename in enumerate(filenames):
            _append_file(images, filename, endswith, startswith, dirname, only_names, remove_root_folder, path)
        if not recurse:
            break
    return images


def upper_directory(path: str, steps: int = 1) -> str:
    """
    Returns path to directory, to which the given path belongs

    Examples:
        >>> upper_directory('/home/danylo/test/file.zip')
        '/home/danylo/test'

        >>> upper_directory('/home/danylo/test/file.zip', 2)
        '/home/danylo'

        >>> upper_directory('/home/danylo/test/', 1)
        '/home/danylo'

    Args:
        path: path to a directory or file
        steps: how many subdirectories to move up

    Returns:
        str: path to the parent directory

    """
    path = path.replace('\\', '/')
    if path.endswith('/'):
        path = path[:-1]
    path = os.path.split(path)
    assert steps < len(path), 'You are trying to do too many steps'
    if steps > 0:
        path = path[:-steps]
    return os.path.join(*path)


def makedirs2file(filename: str) -> str:
    """
    Creates all folders up to the desired file path

    Args:
        filename: path where the file will be saved in the future

    Returns:
        str: `filename`

    """
    os.makedirs(upper_directory(filename, steps=1), exist_ok=True)
    return filename


def string_hash(string: str, n_letters: int = 9, return_int: bool = False, use_numbers: bool = True,
                use_latin: bool = True) -> Union[int, str]:
    """
    Deterministically hashes the given string

    Args:
        string: str to be hashed
        n_letters: how many letters to use for hashing
        return_int: whether to return int or string
        use_numbers: whether to use numbers when turning the hash into string
        use_latin: whether to use latin alphabet when turning the hash into string

    Returns:
        int | str: depends on what the value of return_int is, returns the hash in form of int or string

    """
    _alphabet = []
    if use_numbers:
        _alphabet += list(range(ord('0'), ord('9') + 1))
    if use_latin:
        _alphabet += list(range(ord('a'), ord('z') + 1))

    byte_string = string.encode()
    hashed = sha512(byte_string)
    hashed.update(byte_string)
    digest = hashed.digest()
    integer = 0
    for i, byte in enumerate(digest):
        integer += byte << (i * 8)
    if return_int or len(_alphabet) == 0:
        return integer
    hashed_letters = []
    while integer > 0:
        idx = integer % len(_alphabet)
        letter = chr(_alphabet[idx])
        integer //= len(_alphabet)
        hashed_letters.append(letter)
    return ''.join(hashed_letters)[:n_letters]
