import os


def verify_output_path(file):
    d = os.path.split(file)[0]
    if not os.path.exists(d):
        os.makedirs(d)
