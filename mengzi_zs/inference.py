from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from .prompts import create_input_with_prompt


class MengziZeroShot(object):
    def __init__(self):
        pass

    # def prompt_map(self, task_type, input_string, label_list=None):
    def load(self, model_name="Langboat/mengzi-t5-base-mt"):
        # load model
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name)

    def token_decode(self, s):
        return self.tokenizer.decode(s, skip_special_tokens=True)

    def pick_most_common(self, x: list) -> str:
        return Counter(x).most_common(1)[0][0]

    def inference(self, task_type, input_string, input_string2=None, entity1=None, entity2=None):
        # make input text
        input_text = create_input_with_prompt(
            task_type, input_string, input_string2, entity1, entity2)
        # tokenize
        encodings = self.tokenizer(
            input_text, max_length=512, padding=True, return_tensors="pt")
        # print('encodings', encodings)

        # model inference
        outputs = self.model.generate(
            encodings['input_ids'], attention_mask=encodings['attention_mask'], max_length=512, num_beams=1)
        # print('outputs', outputs)

        # decode model output
        dec_out = list(map(self.token_decode, outputs))

        # print("dec_out: ", dec_out)
        # return result to web
        return self.pick_most_common(dec_out)
