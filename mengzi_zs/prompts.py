# make inputs with prompt

class PromptMap():
    def __init__(self):
        pass

    def task_type_map(self, task_type):
        task_map = {
                    'sentiment_classifier': self.sentiment_cls,
                    'news_classifier': self.news_cls,
                    "medical_domain_intent_classifier":self.domain_cls,
                    "entity_extraction":self.entity_extr,
                    'text_similarity': self.text_sim,
                    'financial_relationship_extraction':self.finance_extr,
                    'ad_generation':self.ad_gen,
                    'comment_object_extraction':self.com_obj_extr,
                    }

        return task_map[task_type]

    def create_input_with_prompt(self, task_type, input_string, input_string_2=None):
        self.prompt_map = self.task_type_map(task_type)

        if task_type == 'text_similarity':
            return self.prompt_map(input_string,input_string_2)
        return self.prompt_map(input_string)

    # map task type
    def sentiment_cls(self, s):
        # dataset name: eprstmt
        # 评论情感分类： 消极/积极 
        prompts = [f'评论:{s}。请判断该条评论所属类别(积极或消极)并填至空格处。回答：']
                #    f'"{s}"。 如果这个评论的作者是客观的，那么请问这个评论的内容是什么态度的回答？答：',
                #    f'现有机器人能判断句子是消极评论还是积极评论。已知句子：“{s}”。这个机器人将给出的答案是：'
        return prompts 

    def news_cls(self, s):
        # dataset name: tnews
        # 新闻分类
        label_list = ['故事', '文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '股票', '农业', '电竞']
        
        prompts = [f'“{s}”是什么新闻频道写的？选项：{"，".join(label_list)}。答：',]
                #    f'这条新闻是关于什么主题的？新闻：{s}。选项：{"，".join(label_list)}。答：',
                #    f'这是关于“{"，".join(label_list)}”中哪个选项的文章？文章：{s}。 答：']
        return prompts
    

    def domain_cls(self,s):
        prompts = [f'问题:“{s}”。此问题的医学意图是什么？选项：病情诊断，病因分析，治疗方案，就医建议，指标解读，疾病描述，后果表述，注意事项，功效作用，医疗费用。']
        return prompts
    
    def entity_extr(self,s):
        prompts = [f'“{s}”找出上述句子中的实体和他们对应的类别']
        return prompts
    
    def text_sim(self,s1,s2):
        prompts = [f'“{s1}”和“{s2}”这两句话是在说同一件事吗?']
        return prompts

    def ad_gen(self,s):
        prompts = [f'请根据以下产品信息设计广告文案。商品信息:{s}']
        return prompts
    
    def finance_extr(self,s):
        prompts = [f'{s}答:']
        return prompts
    
    def com_obj_extr(self,s):
        prompts = [f'评论:{s}.这条评论的评价对象是谁？']
        return prompts


