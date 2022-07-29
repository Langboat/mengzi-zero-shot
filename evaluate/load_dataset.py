import pandas as pd
import json

def tnews_dataset()-> pd.DataFrame():
    filename1 = '../datasets/tnews/dev.json'
    filename2 = '../datasets/tnews/label_index2en2zh.json'

    def _read_json(input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
        return [json.loads(line.strip()) for line in reader] 

    label2zh = {}
    with open(filename2, "r") as f:
        reader = f.readlines()
        for line in reader:
            t = json.loads(line.strip())
            i, label_zh = t['label'], t['label_zh']
            label2zh[i] = label_zh

    df = pd.DataFrame.from_records(_read_json(filename1))
    df['label'] = df['label'].apply(lambda x: label2zh[str(x)])
    df['input_string'] = df['sentence']
    df = df[['input_string', 'label']]
    return df