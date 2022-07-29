from mengzi_zs import MengziZeroShot

mz = MengziZeroShot()
mz.load()

# 使用示例：
# 情感分类
task_type = 'sentiment_classifier'
input_string = '15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错'
res = mz.inference(task_type=task_type, input_string=input_string)

print(f'task_type:{task_type}')
print(f'input_string:{input_string}')
print(f'result:{res}')
print()

# 新闻分类
task_type ='news_classifier'
input_string = '商赢环球股份有限公司关于延期回复上海证券交易所对公司2017年年度报告的事后审核问询函的公告'
res = mz.inference(task_type=task_type, input_string=input_string)

print(f'task_type:{task_type}')
print(f'input_string:{input_string}')
print(f'result:{res}')
print(res)
print()