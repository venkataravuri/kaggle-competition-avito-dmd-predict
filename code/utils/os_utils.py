import os


def _create_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
