import json
import time


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


def save_json(input_json):
    VERSION = time.strftime("%Y%m%d-%H%M%S")
    with open(f'evaluate/results/result-{VERSION}.json', 'w', encoding='utf-8') as f:
        json.dump(input_json, f, ensure_ascii=False, indent=4)
    print(f'result-{VERSION}.json save completely..')
