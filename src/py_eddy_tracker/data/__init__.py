from os import path


def get_path(name):
    return path.join(path.dirname(__file__), name)
