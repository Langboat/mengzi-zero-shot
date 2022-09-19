import pandas as pd
import json
from utils import read_json, read_txt, text_processing


def tnews_dataset() -> pd.DataFrame():
    filename1 = './datasets/tnews/dev.json'
    filename2 = './datasets/tnews/label_index2en2zh.json'

    label2zh = {}
    with open(filename2, "r") as f:
        reader = f.readlines()
        for line in reader:
            t = json.loads(line.strip())
            i, label_zh = t['label'], t['label_zh']
            label2zh[i] = label_zh

    df = pd.DataFrame.from_records(read_json(filename1))
    df['label'] = df['label'].apply(lambda x: label2zh[str(x)])
    df['input_string'] = df['sentence']
    df = df[['input_string', 'label']]
    return df


def eprstmt_dataset() -> pd.DataFrame():
    filename = './datasets/eprstmt/dev.json'

    label2zh = {'Positive': '积极',
                'Negative': '消极'}

    df = pd.DataFrame.from_records(read_json(filename))
    df['label'] = df['label'].apply(lambda x: label2zh[str(x)])
    df['input_string'] = df['sentence']
    df = df[['input_string', 'label']]
    return df


def lcqmc_dataset() -> pd.DataFrame():
    filename = './datasets/lcqmc/dev.txt'
    label2zh = {'0': '否', '1': '是'}

    df = pd.DataFrame.from_records(read_txt(filename), columns=[
                                   "input_string", "input_string2", "label"])
    df['label'] = df['label'].apply(lambda x: label2zh[str(x.strip())])
    df['input_string'] = df['input_string'].apply(
        lambda x: text_processing(x))
    df['input_string2'] = df['input_string2'].apply(
        lambda x: text_processing(x))
    df = df[['input_string', 'input_string2', 'label']]
    return df


def cluner_dataset() -> pd.DataFrame():
    filename = './datasets/cluner/dev.json'
    # 地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

    label2zh = {'address': '地址',
                'book': '书名',
                'company': '公司',
                'game': '游戏',
                'goverment': '政府',
                'movie': '电影',
                'name': '姓名',
                'organization': '组织',
                'position': '职位',
                'scene': '景点'
                }

    def _label(label):
        for x in label.keys():
            if x in label2zh.keys():
                label[label2zh[x]] = list(label.pop(x).keys())[0]

        return label

    df = pd.DataFrame.from_records(read_json(filename))
    df['label'] = df['label'].apply(lambda x: _label(x))
    df['input_string'] = df['text']
    df = df[['input_string', 'label']]

    return df


def finre_dataset() -> pd.DataFrame():
    filename = './datasets/finre/valid.txt'
    # label2zh={'0': '否', '1':'是' }

    df = pd.DataFrame.from_records(read_txt(filename), columns=[
                                   "entity1", "entity2", "label", "sentence"])
    # df.replace(to_replace='None', value=np.nan).dropna()
    df = df.dropna()

    df['label'] = df['label'].apply(lambda x: str(x.strip()))
    df['entity1'] = df['entity1'].apply(lambda x: text_processing(x))
    df['entity2'] = df['entity2'].apply(lambda x: text_processing(x))
    df['sentence'] = df['sentence'].apply(lambda x: text_processing(x))
    df['input_string'] = df['sentence']

    df = df[['input_string', 'entity1', 'entity2', 'label']]
    return df


def cote_dataset() -> pd.DataFrame():
    """
    由于Cote数据集dev数据集没有label,所以在这里采用train数据集。
    """
    filename_list = ['./datasets/cote/COTE-BD/train.tsv',
                     './datasets/cote/COTE-DP/train.tsv',
                     './datasets/cote/COTE-MFW/train.tsv']
    # label2zh={'0': '否', '1':'是' }
    df_all = pd.DataFrame()

    def _read_txt(input_file):
        with open(input_file, encoding="utf-8") as f:
            reader = f.readlines()
            return [line.split('\t', 1) for line in reader[1:]]

    for filename in filename_list:
        df = pd.DataFrame.from_records(_read_txt(filename), columns=[
                                       "label", "input_string"])
        df = df.dropna()

        df['label'] = df['label'].apply(lambda x: str(x.strip()))
        df['input_string'] = df['input_string'].apply(
            lambda x: text_processing(x))

        df = df[['input_string', 'label']]
        df_all = pd.concat([df_all, df], axis=0)
        df_all.sample(n=32, replace=True, random_state=42)
    return df_all


def cepsum_dataset() -> pd.DataFrame():
    filename = './datasets/cepsum/valid.json'

    def _text_processing(text):
        res = ['\ufeff', '\xa0', '\u3000']
        # if isinstance(text,list):
        text_prcessed = []
        for i in range(len(text)):
            tmp = text[i].strip()
            for ch in res:
                tmp = tmp.replace(ch, '')
            text_prcessed.append(tmp)

        return text_prcessed

    df = pd.DataFrame.from_records(read_json(filename))
    df = df.dropna()

    df['label'] = df['targets'].apply(lambda x: _text_processing(x))
    df['input_string'] = df['source'].apply(lambda x: _text_processing(x))

    df = df[['input_string', 'label']]
    return df


def kuake_qic_dataset() -> pd.DataFrame():
    filename = './datasets/kuake-qic/processed_KUAKE-QIC_dev.json'
    df = pd.DataFrame.from_records(read_json(filename))

    df['label'] = df['label'].apply(lambda x: text_processing(x))
    df['input_string'] = df['query'].apply(lambda x: text_processing(x))
    df = df[['input_string', 'label']]
    return df


def express_ner_dataset() -> pd.DataFrame():
    filename = './datasets/express-ner/dev.txt'
    data = open(filename).readlines()
    data = data[1:]
    data = [line.strip().split('\t') for line in data]
    new_data = []
    for ins in data:
        text = ins[0]
        gold = ins[1]
        text = text.split('\x02')
        text = ''.join(text)
        gold = gold.split('\x02')
        name_start = None
        name_end = None
        for i, c in enumerate(gold):
            if c == 'B-P':
                name_start = i
            elif c == 'I-P':
                if i == len(gold) - 1:
                    name_end = i
                elif gold[i+1] != 'I-P':
                    name_end = i
        if name_start is None or name_end is None:
            continue
        new_data.append({'input_string': text_processing(text), 'label': text_processing(text[name_start:name_end+1])})
    df = pd.DataFrame(new_data)
    return df

