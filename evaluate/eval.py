# from mengzi_zs import MengziZeroShot

from load_data import eprstmt_dataset, tnews_dataset, lcqmc_dataset, cluner_dataset, finre_dataset, cote_dataset, cepsum_dataset, quake_qic_dataset
from metrics import cal_acc, ner_get_f1_score, cal_f1, rouge_2_corpus_multiple_target
import sys
sys.path.append('./')
from mengzi_zs import MengziZeroShot

datasets = {'sentiment_classifier': eprstmt_dataset(),
            'news_classifier': tnews_dataset(),
            'text_similarity': lcqmc_dataset(),
            'entity_extraction': cluner_dataset(),
            "financial_relationship_extraction": finre_dataset(),
            "comment_object_extraction": cote_dataset(),
            "ad_generation": cepsum_dataset(),
            "medical_domain_intent_classifier": quake_qic_dataset()}

mp = MengziZeroShot()  # default
mp.load()

# for test
DEV_NUM = 32

task_name_list = list(datasets.keys())

for task_name in task_name_list:

    df = datasets[task_name]
    df = df[:DEV_NUM]

    res_list = []
    for s in list(df['input_string']):
        res = mp.inference(task_type=task_name, input_string=s)
        res_list.append(res)
    df['pred'] = res_list

    if task_name in ['sentiment_classifier', 'news_classifier', 'text_similarity', "medical_domain_intent_classifier"]:
        res_score = cal_acc(labels=list(df['label']), preds=list(df['pred']))
        print(task_name, f"acc: {res_score}")

    elif task_name in ["entity_extraction"]:
        cvt_res = []
        for res in list(df['pred']):
            res = res.split(',')
            sub_cvt_res = {}
            for sub_res in res:
                content, entity = sub_res.split(":")
                sub_cvt_res[entity] = content
            cvt_res.append(sub_cvt_res)
        _, avg = ner_get_f1_score(pre_lines=cvt_res, gold_lines=list(df['label']),
                                  labels=['地址', '书名', '公司', '游戏', '政府', '电影', '姓名', '组织', '职位', '景点'])
        print(task_name, f"avg: {avg}")

    elif task_name in ["financial_relationship_extraction"]:
        res_score = cal_f1(labels=list(df['label']), preds=list(df['pred']))
        print(task_name, f"f1: {res_score}")

    elif task_name in ["comment_object_extraction"]:
        res_score = cal_f1(labels=list(df['label']), preds=list(df['pred']))
        print(task_name, f"f1: {res_score}")

    elif task_name in ["ad_generation"]:
        res_score = rouge_2_corpus_multiple_target(peers=list(df['label']), models=list(df['pred']))
        print(task_name, f"rouge_2: {res_score}")

    else:
        raise ValueError("not implement!")
