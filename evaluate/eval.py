from load_data import *
from metrics import cal_acc
from ../mengzi_zs import MengziZeroShot

#  Overview
#  task: sentiment_classifier, dataset: eprstmt
#  task: news_classifier, dataset: tnews

datasets = {'sentiment_classifier': eprstmt_dataset(),
            'news_classifier': tnews_dataset()}

mp = MengziZeroShot()  # default
mp.load()

task_name_list = ['sentiment_classifier',
                  'news_classifier']

# for test
DEV_NUM = 32

for task_name in task_name_list:
    res_list = []
    df = datasets[task_name]
    for s in list(df['input_string'])[:DEV_NUM]:
        res = mp.inference(task_type=task_name,
                           input_string=s)
        res_list.append(res)
    res_score = cal_acc(list(df['label'])[:DEV_NUM], res_list)
    print(task_name, res_score)
