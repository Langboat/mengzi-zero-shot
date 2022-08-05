from load_data import *
from metrics import cal_acc,ner_get_f1_score,cal_f1,rouge_1_corpus,rouge_2_corpus_multiple_target
import sys
sys.path.append("../")
from mengzi_zs import MengziZeroShot
# from ../mengzi_zs import MengziZeroShot
#  Overview
#  task: sentiment_classifier, dataset: eprstmt
#  task: news_classifier, dataset: tnews

datasets = {'sentiment_classifier': eprstmt_dataset(),
            'news_classifier': tnews_dataset(),
            'text_similarity':lcqmc_dataset(),
            'entity_extraction':cluner_dataset(),
            "financial_relationship_extraction":finre_dataset(),
            "comment_object_extraction":cote_dataset(),
            "ad_generation":cepsum_dataset(),
            "medical_domain_intent_classifier":quake_qic_dataset()}

mp = MengziZeroShot()  # default
mp.load()

task_name_list = list(datasets.keys())
# for test
DEV_NUM = 32

for task_name in task_name_list:
    
    res_list = []
    df_label_list=[]
    df = datasets[task_name]
    
    if isinstance(df,list):
        for sub_df in df:
            sub_res=[]
            for s in list(sub_df['input_string'])[:DEV_NUM]:
                res = mp.inference(task_type=task_name,input_string=s)
                sub_res.append(res)  
            res_list.append(sub_res)
            df_label_list.append(list(sub_df['label'])[:DEV_NUM])
    
    else:
        for s in list(df['input_string'])[:DEV_NUM]:
            res = mp.inference(task_type=task_name,
                            input_string=s)
            res_list.append(res)  


    if task_name in ['sentiment_classifier','news_classifier','text_similarity',"medical_domain_intent_classifier"]:
        res_score = cal_acc(labels=list(df['label'])[:DEV_NUM], preds=res_list)
        print(task_name, f"acc:{res_score}")
    elif task_name == "entity_extraction":
        cvt_res=[]
        for res in res_list:
            res=res.split(',')
            sub_cvt_res={}
            
            for sub_res in res:
                content,entity = sub_res.split(":")
                sub_cvt_res[entity]=content
            cvt_res.append(sub_cvt_res)

        # print(cvt_res)
        f_score, avg = ner_get_f1_score(pre_lines=cvt_res, gold_lines=list(df['label'])[:DEV_NUM])
        print(task_name, f"avg:{avg} f_score:{f_score}" )
    elif task_name in ["financial_relationship_extraction"]:
        res_score = cal_f1(labels=list(df['label'])[:DEV_NUM], preds=res_list)
        print(task_name, f"f1:{res_score}")
    elif task_name in ["comment_object_extraction"]:
        sum_f1=[]
        for preds,labels in zip(res_list,df_label_list):
            res_score = cal_f1(labels=labels, preds=preds)
            sum_f1.append(res_score)
            avg_f1=sum(sum_f1)/len(sum_f1)
        print(task_name, f"avf f1:{avg_f1}")
    elif task_name in ["ad_generation"]:
        res_score = rouge_2_corpus_multiple_target(peers=list(df['label'])[:DEV_NUM], models=res_list)
        print(task_name, f"rouge_2:{res_score}")

    else:
        raise ValueError("not implement!")