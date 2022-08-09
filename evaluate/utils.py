import json


def text_processing(text):
    res = ['\ufeff', '\xa0', '\u3000']
    text = text.strip()
    for ch in res:
        text = text.replace(ch, '')
    return text


def read_txt(input_file):
    with open(input_file, encoding="utf-8") as f:
        reader = f.readlines()
        return [tuple(line.split('\t')) for line in reader]


def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r") as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]
