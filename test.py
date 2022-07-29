from mengzi_zs import MengziZeroShot

mz = MengziZeroShot()
mz.load()

# 使用示例：

# 单条测试
task_type = 'sentiment_classifier'
input_string = '15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错'
res = mz.inference(task_type=task_type, input_string=input_string)

print(f'task_type:{task_type}')
print(f'input_string:{input_string}')
print(f'result:{res}')
print()


# 批量测试 
inputs = [# 实体抽取
          {'task_type': 'entity_extractor', 
           'input_string':'导致泗水的砭石受到追捧，价格突然上涨。而泗水县文化市场综合执法局颜鲲表示，根据监控，'},
          # 语义相似度
          {'task_type': 'text_similarity', 
           'input_string':'你好，我还款银行怎么更换',
           'input_string_2':'怎么更换绑定还款的卡',
           },
            # 情感分类
            {'task_type': 'sentiment_classifier', 
            'input_string':'15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错'},
           # 新闻分类
            {'task_type': 'sentiment_classifier', 
            'input_string':'差得要命,很大股霉味,勉强住了一晚,第二天大早赶紧溜'}
           
            ]

for t in inputs:
    task_type = t['task_type']
    input_string = t['input_string']
    res = mz.inference(task_type=task_type, input_string=input_string)

    print(f'task_type:{task_type}')
    print(f'input_string:{input_string}')
    print(f'result:{res}')
    print()
