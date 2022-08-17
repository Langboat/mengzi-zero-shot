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

## 接口能力

| 接口能力                 | 简要描述                                             |
| ---------------- | ------------------------------------------------------------ |
| <a href="#entity">实体抽取</a> | 抽取文本中的命名实体，并提供对应实体的类别，如地址、书名、公司、游戏、政府、电影、姓名、组织、职位、景点等等。 |
| <a href="#similarity">语义相似度</a>       | 衡量两个文本之间的语义相似性。                                       |
| <a href="#finance">金融关系抽取</a>   | 判断文本中的两个实体属于哪种关系，如发行、被发行、注资、增持等等。                           |
| <a href="#ad">广告文案生成</a>      | 给定商品的关键词信息，生成合理的广告文案。                     |
| <a href="#medical">医学领域意图分类</a>  | 根据输入的医疗领域文本，识别用户查询意图，如指标解读、病情诊断、就医建议、治疗方案等等。                         |
| <a href="#sentiment">情感分类</a>          | 对文本进行情感极性类别分类（积极、消极）。 |
| <a href="#obj">评论对象抽取</a>          | 对于给定的评论文本，自动抽取其中包含的评价对象。                 |
| <a href="#news">新闻分类</a>          | 对输入的新闻文本进行分类，如农业、文化、电竞、体育、财经、娱乐、旅游、教育、金融、军事、房产、汽车、股票、国际等等。 |

### 接口说明

### <span id="entity"> 实体抽取 </span>

输入一段文本，抽取文中的实体，并判定其所属类别，如地址、书名、公司、游戏、政府、电影、姓名、组织、职位、景点等等。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='entity_extraction', 
                   input_string='导致泗水的砭石受到追捧，价格突然上涨。而泗水县文化市场综合执法局颜鲲表示，根据监控')
print(res)
```

#### 模型输出结果

```
"泗水：地址，泗水县文化市场综合执法局：政府，颜鲲：姓名"
```

### <span id="similarity"> 语义相似度 </span>

输入两段文本，判断其语义是否相同。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='text_similarity', 
                   input_string='你好，我还款银行怎么更换',
                   input_string2='怎么更换绑定还款的卡')
print(res)
```

#### 模型输出结果
```
"是"
```

### <span id="finance"> 金融关系抽取 </span>


输入一段文本，以及文本中蕴含的两个实体，判断文本中的两个实体属于哪种关系。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='financial_relationship_extraction', 
                input_string='为打消市场顾虑,工行两位洋股东——美国运通和安联集团昨晚做出承诺,近期不会减持工行H股。',
                entity1='工行',
                entity2='美国运通')
print(res)
```

#### 模型输出结果
```
 "被持股"
```

### <span id="ad"> 广告文案生成 </span>

输入一段商品的描述信息文本，自动生成有效的广告文案。

```python
 from mengzi_zs import MengziZeroShot
 mz = MengziZeroShot()
 mz.load()
 res = mz.inference(task_type='ad_generation', 
                    input_string='类型-裤，版型-宽松，风格-潮，风格-复古，风格-文艺，图案-复古，裤型-直筒裤，裤腰型-高腰，裤口-毛边')
print(res)
```

#### 模型输出结果
```
 "小宽松版型与随性的风格颇具人气，高腰的设计，毛边裤脚，增添潮流气息。考究的做旧质感，洋溢着复古的气息，一款风格随性却不失复古文艺的直筒牛仔裤。,宽松的直筒版型，对身材的包容度较大，穿着舒适无束缚感。高腰的设计，提升腰线，拉长身材比例，打造大长腿的既视感"
```

### <span id="medical"> 医学领域意图分类 </span>

输入一段医疗查询文本，识别出文本中用户的查询意图。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='medical_domain_intent_classifier', 
                input_string='呼气试验阳性什么意思')
print(res)
```
#### 模型输出结果

```
 "指标解读"
```

### <span id="sentiment"> 情感分类 </span> 

输入一段文本，对包含主观观点信息的文本进行情感极性类别分类（积极、消极）。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='sentiment_classifier', 
                input_string='房间很一般，小，且让人感觉脏，隔音效果差，能听到走廊的人讲话，走廊光线昏暗，旁边没有什么可吃')
print(res)
```
#### 模型输出结果
```
 "消极"
```

### <span id="obj"> 评论对象抽取 </span> 

输入一段文本，自动抽取其中包含的评价对象。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='comment_object_extraction', 
                input_string='灵水的水质清澈，建议带个浮潜装备，可以看清湖里的小鱼。')
print(res)
```
#### 模型输出结果

```
 "灵水"
```

### <span id="news"> 新闻分类 </span> 

输入一段新闻，识别其所属类别，如农业、文化、电竞、体育、财经、娱乐、旅游、教育、金融、军事、房产、汽车、股票、国际等等。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='news_classifier', 
                   input_string='懒人适合种的果树：长得多、好打理，果子多得都得送邻居吃')
print(res)
```
#### 模型输出结果
```
 "农业"
```

##  代码格式规范

```
pip install flake8==5.0.4
# usage: 
flake8 --ignore=E501,E402 test.py
```

## TODO

1. 目前 cpu inference ，可补充gpu inference
2. 对比 cpu gpu inference 结果差异


