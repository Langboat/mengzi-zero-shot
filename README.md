<h2 align="center">Mengzi Zero-Shot</h2>

<div align="center">
    <a href="https://github.com/Langboat/mengzi-zero-shot/actions">
       <img alt="Unit Tests" src="https://github.com/Langboat/mengzi-zero-shot/actions/workflows/unit-tests.yml/badge.svg?branch=main">
    </a>
    <a href="https://pypi.org/project/mengzi-zero-shot/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/mengzi-zero-shot?color=blue">
    </a>
    <a href="https://pypi.org/project/mengzi-zero-shot/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/mengzi-zero-shot?colorB=blue">
    </a>
</div>

本项目提供的所有接口能力基于 [Mengzi-T5-base-MT](https://huggingface.co/Langboat/mengzi-t5-base-mt)。该模型是一个多任务模型，是在 [Mengzi-T5-base](https://huggingface.co/Langboat/mengzi-t5-base) 的基础上，使用了额外的27个数据集及301个 prompt 进行了多任务训练得到的。本项目提供实体抽取、语义相似度、金融关系抽取、广告文案生成、医学领域意图分类、情感分类、评论对象抽取、新闻分类等能力，开箱即用。


# 导航
* [快速上手](#快速上手)
* [接口能力](#接口能力)
* [接口说明](#接口说明)
* [贡献代码](#贡献代码)
* [联系方式](#联系方式)
* [免责声明](#免责声明)

# 快速上手
## 新建环境
```
conda create -n mengzi_env python=3.7 -y
conda activate mengzi_env
```

## pip安装
```bash
pip install mengzi-zero-shot
```

## 测试样例
```
python test.py
```

# 接口能力

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
| <a href="#name">人名抽取</a> | 抽取文本中的人名。 |
| <a href="#company">公司名抽取</a> | 抽取文本中的公司名。 |


# 接口说明

## <span id="entity"> 实体抽取 </span>

输入一段文本，抽取文中的实体，并判定其所属类别，如地址、书名、公司、游戏、政府、电影、姓名、组织、职位、景点等等。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='entity_extraction', 
                   input_string='导致泗水的砭石受到追捧，价格突然上涨。而泗水县文化市场综合执法局颜鲲表示，根据监控')
print(res)
```

Output:

```
"泗水：地址，泗水县文化市场综合执法局：政府，颜鲲：姓名"
```

## <span id="similarity"> 语义相似度 </span>

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

Output:
```
"是"
```

## <span id="finance"> 金融关系抽取 </span>


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

Output:
```
 "被持股"
```

## <span id="ad"> 广告文案生成 </span>

输入一段商品的描述信息文本，自动生成有效的广告文案。

```python
 from mengzi_zs import MengziZeroShot
 mz = MengziZeroShot()
 mz.load()
 res = mz.inference(task_type='ad_generation', 
                    input_string='类型-裤，版型-宽松，风格-潮，风格-复古，风格-文艺，图案-复古，裤型-直筒裤，裤腰型-高腰，裤口-毛边')
print(res)
```

Output:
```
 "小宽松版型与随性的风格颇具人气，高腰的设计，毛边裤脚，增添潮流气息。考究的做旧质感，洋溢着复古的气息，一款风格随性却不失复古文艺的直筒牛仔裤。,宽松的直筒版型，对身材的包容度较大，穿着舒适无束缚感。高腰的设计，提升腰线，拉长身材比例，打造大长腿的既视感"
```

## <span id="medical"> 医学领域意图分类 </span>

输入一段医疗查询文本，识别出文本中用户的查询意图。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='medical_domain_intent_classifier', 
                   input_string='呼气试验阳性什么意思')
print(res)
```
Output:

```
 "指标解读"
```

## <span id="sentiment"> 情感分类 </span> 

输入一段文本，对包含主观观点信息的文本进行情感极性类别分类（积极、消极）。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='sentiment_classifier', 
                   input_string='房间很一般，小，且让人感觉脏，隔音效果差，能听到走廊的人讲话，走廊光线昏暗，旁边没有什么可吃')
print(res)
```
Output:
```
 "消极"
```

## <span id="obj"> 评论对象抽取 </span> 

输入一段文本，自动抽取其中包含的评价对象。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='comment_object_extraction', 
                   input_string='灵水的水质清澈，建议带个浮潜装备，可以看清湖里的小鱼。')
print(res)
```
Output:

```
 "灵水"
```

## <span id="news"> 新闻分类 </span> 

输入一段新闻，识别其所属类别，如农业、文化、电竞、体育、财经、娱乐、旅游、教育、金融、军事、房产、汽车、股票、国际等等。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='news_classifier', 
                   input_string='懒人适合种的果树：长得多、好打理，果子多得都得送邻居吃')
print(res)
```
Output:
```
 "农业"
```

## <span id="name"> 人名抽取 </span>

输入一段文本，识别其中出现的人名。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='name_extraction',
		   input_string='我是张三，我爱北京天安门')
print(res)
```
Output:
```
"张三"
```

## <span id="company"> 公司名抽取 </span>

输入一段文本，识别其中出现的公司名。

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='company_extraction',
		   input_string='就天涯网推出彩票服务频道是否是业内人士所谓的打政策“擦边球”，记者近日对此事求证彩票监管部门。')
print(res)
```
Output:
```
"天涯网:公司"
```

# 贡献代码

## 为本项目改善 prompt
可以在`mengzi_zs/prompts.py`查看已有 SDK 功能使用的相应的prompt，如有在已有任务对应的 dev 数据集上获得更好分数的 prompt，欢迎提交 issue 或直接提 pull request。分数以`evaluate/eval.py`执行结果为准。

## 为本项目贡献新功能
需要：
1. `mengzi_zs/prompts.py`中构建新功能函数
2. `datasets`文件夹中添加对应新功能的数据集
3. `evaluate/eval.py`中添加测评计算方式
欢迎提交 issue 或直接提 pull request，分数以`evaluate/eval.py`执行结果为准。

## 提交代码的格式规范
```
pip install flake8==5.0.4
flake8 --ignore=E501,E402 test.py
```


# 联系方式

## 微信讨论群
<img width="200" alt="image" src="https://user-images.githubusercontent.com/26166111/185085965-4ebefc47-ede6-47c8-9494-ad645a18e067.jpg">



## 邮箱
huajingyun[at]langboat[dot]com


# 免责声明
该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。 实验结果可能因随机数种子，计算设备而发生改变。

使用者以各种方式使用本模型（包括但不限于修改使用、直接使用、通过第三方使用）的过程中，不得以任何方式利用本模型直接或间接从事违反所属法域的法律法规、以及社会公德的行为。使用者需对自身行为负责，因使用本模型引发的一切纠纷，由使用者自行承担全部法律及连带责任。我们不承担任何法律及连带责任。

我们拥有对本免责声明的解释、修改及更新权。
