import sys
import time
from load_data import eprstmt_dataset, tnews_dataset, lcqmc_dataset, cluener_dataset, finre_dataset, cote_dataset, cepsum_dataset, kuake_qic_dataset, express_ner_dataset
from metrics import cal_acc, ner_get_f1_score, cal_f1, rouge_n_corpus_multiple_target
from utils import save_json
sys.path.append('./')
from mengzi_zs import MengziZeroShot

import pandas as pd
pd.options.mode.chained_assignment = None

datasets = {'sentiment_classifier': eprstmt_dataset(),
            'news_classifier': tnews_dataset(),
            'text_similarity': lcqmc_dataset(),
            'entity_extraction': cluener_dataset(),
            "financial_relationship_extraction": finre_dataset(),
            "comment_object_extraction": cote_dataset(),
            "ad_generation": cepsum_dataset(),
            "medical_domain_intent_classifier": kuake_qic_dataset(),
            'name_extraction': express_ner_dataset(),
            'company_extraction': cluener_dataset()}

mp = MengziZeroShot()  # default
mp.load()

# for test
DEV_NUM = 32

task_name_list = list(datasets.keys())
results = {}
time_start = time.time()
for task_name in task_name_list:

    df = datasets[task_name]

    if task_name in ['company_extraction']:
        df = df[:DEV_NUM*6]
    else:
        df = df[:DEV_NUM]

    # inference
    if task_name in ['text_similarity']:
        for i, row in df.iterrows():
            res = mp.inference(task_type=task_name,
                               input_string=row[0], input_string2=row[1])
            df.loc[i, 'pred'] = res
    elif task_name in ["financial_relationship_extraction"]:
        for i, row in df.iterrows():
            res = mp.inference(
                task_type=task_name, input_string=row[0], entity1=row[1], entity2=row[2])
            df.loc[i, 'pred'] = res
    else:
        for i, row in df.iterrows():
            res = mp.inference(task_type=task_name, input_string=row[0])
            df.loc[i, 'pred'] = res

    results[task_name] = {}

    if task_name in ['sentiment_classifier', 'news_classifier', 'text_similarity', "medical_domain_intent_classifier"]:
        res_score = cal_acc(labels=list(df['label']), preds=list(df['pred']))
        print(task_name, f"acc: {res_score}")
        results[task_name]['acc'] = res_score

    elif task_name in ["entity_extraction"]:
        cvt_res = []
        for res in list(df['pred']):
            res = res.split(',')
            sub_cvt_res = {}
            for sub_res in res:
                content, entity = sub_res.split(":")
                sub_cvt_res[entity] = content
            cvt_res.append(sub_cvt_res)
        _, f1_score_avg = ner_get_f1_score(pre_lines=cvt_res, gold_lines=list(df['label']), labels=['地址', '书名', '公司', '游戏', '政府', '电影', '姓名', '组织', '职位', '景点'])
        print(f"{task_name} f1_score_avg: {f1_score_avg}")
        results[task_name]['f1_score_avg'] = f1_score_avg

    elif task_name in ["financial_relationship_extraction", "comment_object_extraction", "name_extraction"]:
        res_score = cal_f1(labels=list(df['label']), preds=list(df['pred']))
        print(f"{task_name}  f1: {res_score}")
        results[task_name]['f1_score'] = res_score

    elif task_name in ["ad_generation"]:
        res_score = rouge_n_corpus_multiple_target(peers=list(df['label']), models=list(df['pred']))
        print(f"{task_name} rouge_1: {res_score}")
        results[task_name]['rouge_1'] = res_score

    elif task_name in ["company_extraction"]:
        labels = []
        for i in list(df['label']):
            try:
                labels.append(i['公司'])
            except KeyError:
                labels.append('')
        res_score = cal_f1(labels=labels, preds=list(df['pred']))
        print(f"{task_name}  f1: {res_score}")
        results[task_name]['f1_score'] = res_score

    else:
        raise ValueError("not implement!")

save_json(results)
print(f'cost time: {round(time.time()-time_start, 2)} s')
