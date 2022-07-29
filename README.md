# mengzi-zero-shot
NLU &amp; NLG (zero-shot) depend on mengzi-t5-base-mt pretrained model

## Quick Start
### 新建 conda 环境
```
conda create -n mengzi_env python=3.7.11 -y
conda activate mengzi_env
```

### pip 安装 （release后这行会改） 
```bash
pip install mengzi_zero_shot-1.0.0-py3-none-any.whl
# pip install mengzi-zero-shot
```

### 测试提供的样例数据
```
python test.py
```


### 已有功能
```
# 实体抽取, entity_extraction
# 语义相似度, text_similarity
# 金融关系抽取, financial_relationship_extraction
# 广告文案生成, ad_generation
# 医学领域意图分类, medical_domain_intent_classifier
# 情感分类, sentiment_classifier, dataset: eprstmt
# 对象抽取, comment_object_extraction
# 新闻分类, news_classifier, dataset: tnews
# 

```

### 使用features
```python
from mengzi_zs import MengziZeroShot

mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='sentiment_classifier', 
                   input_string='15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错')
```

可能遇到问题：
```
# 由于目前是 huggingface 上 private 状态，联系我关联公司账号组后，执行如下指令
transformers-cli login 
自行根据提示进行账号关联
方可顺利使用我们的private模型
```


## TODO
1. 目前 cpu inference ，可补充gpu inference
2. 对比 cpu gpu inference 结果差异
