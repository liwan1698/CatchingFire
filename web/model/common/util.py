import yaml
import logging
from tqdm import tqdm


def read_yaml(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    cont = f.read()
    return yaml.load(cont)

