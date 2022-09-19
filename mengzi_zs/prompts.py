# make inputs with prompt

def task_type_map(task_type):
    task_map = {
        'sentiment_classifier': sentiment_cls,
        'news_classifier': news_cls,
        "medical_domain_intent_classifier": domain_cls,
        "entity_extraction": entity_extr,
        'text_similarity': text_sim,
        'financial_relationship_extraction': finance_extr,
        'ad_generation': ad_gen,
        'comment_object_extraction': com_obj_extr,
        'name_extraction': name_extr,
        'company_extraction': entity_extr,
    }

    return task_map[task_type]


def create_input_with_prompt(task_type, input_string, input_string2=None, entity1=None, entity2=None):
    prompt_map = task_type_map(task_type)

    if task_type == 'text_similarity':
        return prompt_map(input_string, input_string2)
    elif task_type == 'financial_relationship_extraction':
        return prompt_map(input_string, entity1, entity2)
    return prompt_map(input_string)


def entity_extr(s,):
    '''
    dataset: CLUENER
    task: 实体抽取
    output:
    '''
    prompts = [f'“{s}”找出上述句子中的实体和他们对应的类别']
    return prompts


def text_sim(s1, s2):
    '''
    dataset:
    task: 语义相似度
    output:
    '''
    prompts = [f'“{s1}”和“{s2}”这两句话是在说同一件事吗?']
    return prompts


def finance_extr(s, e1, e2):
    '''
    dataset:
    task: 金融关系抽取
    output:
    '''
    prompts = [f'“{s}”中的“{e1}”和“{e2}”是什么关系？答:']
    return prompts


def ad_gen(s):
    '''
    dataset:
    task: 广告文案生成
    output:
    '''
    prompts = [f'请根据以下产品信息设计广告文案。商品信息:{s}']
    return prompts


def domain_cls(s):
    '''
    dataset:
    task: 医学领域意图分类
    output:
    '''
    # dataset: quake-qic
    prompts = [
        f'问题:“{s}”。此问题的医学意图是什么？选项：病情诊断，病因分析，治疗方案，就医建议，指标解读，疾病描述，后果表述，注意事项，功效作用，医疗费用。']
    return prompts


def sentiment_cls(s):
    '''
    dataset: eprstmt
    task: 评论情感分类
    output: 消极/积极
    '''
    prompts = [f'评论:{s}。请判断该条评论所属类别(积极或消极)并填至空格处。回答：']
    #    f'"{s}"。 如果这个评论的作者是客观的，那么请问这个评论的内容是什么态度的回答？答：',
    #    f'现有机器人能判断句子是消极评论还是积极评论。已知句子：“{s}”。这个机器人将给出的答案是：'
    return prompts


def com_obj_extr(s):
    '''
    dataset:
    task: 评论对象抽取
    output:
    '''
    prompts = [f'评论:{s}.这条评论的评价对象是谁？']
    return prompts


def news_cls(s):
    '''
    dataset: tnews
    task: 新闻分类
    output:
    '''
    label_list = ['故事', '文化', '娱乐', '体育', '财经', '房产', '汽车',
                  '教育', '科技', '军事', '旅游', '国际', '股票', '农业', '电竞']

    prompts = [f'“{s}”是什么新闻频道写的？选项：{"，".join(label_list)}。答：', ]
    #    f'这条新闻是关于什么主题的？新闻：{s}。选项：{"，".join(label_list)}。答：',
    #    f'这是关于“{"，".join(label_list)}”中哪个选项的文章？文章：{s}。 答：']
    return prompts


def name_extr(s):
    '''
    dataset: express-ner
    task: 人名抽取
    output:
    '''
    prompts = [f'{s}抽取以上句子中的姓名。回答：']
    return prompts


# def company_extr(s):
#     '''
#     dataset: CLUENER
#     task: 公司名抽取
#     output:
#     '''
#     prompts = [f'找出句子“{s}”中实体类别为“公司”的实体。答：', ]
#     return prompts
