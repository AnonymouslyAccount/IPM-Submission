import re
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../../'))
import copy
import json
import torch
import requests
import colorama
from Extraction import utils, config
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


# pipeline_ins = pipeline("text-generation", model=config.Zephyr_7B_model_path, torch_dtype=torch.bfloat16,
#                         device_map="auto")

use_DuIE = False
use_DuEE = False
use_local_model = True
model_path = "/mnt/LLMs/Baichuan2-7B"
call_model = None

if use_local_model:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    # model.generation_config = GenerationConfig.from_pretrained(model_path, temperature=0.1, max_tokens=64,
    #                                                            top_k=10, top_p=0.95)


def call_local_model(prompt, temperature=0.1, max_tokens=128, top_k=10, top_p=0.95, stop='\n\n'):
    model.generation_config = GenerationConfig.from_pretrained(model_path,
                                                               temperature=temperature, max_tokens=max_tokens,
                                                               top_k=top_k, top_p=top_p, stop=stop)
    results = model.chat(tokenizer, [
        {
            'role': 'user',
            'content': prompt
        }
    ])
    return results


def call_remote_model(prompt,  temperature=0.1, max_tokens=128, top_k=10, top_p=0.95, stop='\n\n'):
    results = requests.request('POST', 'http://localhost:8011/generate',
                               json={"prompt": prompt,
                                     "temperature": temperature, "max_tokens": max_tokens,
                                     "top_k": top_k, "top_p": top_p, "stop": stop})
    results = json.loads(results.text)['text'][0]
    results = results.split('<|assistant|>')[-1]
    return results


if use_local_model:
    call_model = call_local_model
else:
    call_model = call_remote_model


def build_model_prompt(content, keep=False, role='你是一个诚实和高效的知识抽取专家'):
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "你是一个诚实和高效的知识抽取专家",
    #     },
    #     {
    #         "role": "user",
    #         "content": content
    #     }
    # ]
    # prompt = pipeline_ins.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(prompt)
    if keep:
        return content
    prompt = \
        "<|system|>\n{role}</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n".format(
            query_str=content)
    # print(prompt)
    # exit()
    return prompt


def NER_template(text, label):
    ner_prompt = """
        给定的句子为："{text}"
        给定实体类型列表：{label}
        请抽取语句中包含的实体类型
        要求按照如下形式回答：
        [实体类型：实体名称]
        输出的时候请不要保留[]，同一类型的多个实体使用'；'隔开，不要输出额外内容
    """
    # 输出的时候请不要保留[]，同一类型的多个实体使用'；'隔开，某个类型没有对应实体则不要输出，
    # 给定的句子为："{}"\n\n给定实体类型列表：{}\n\n在这个句子中，可能包含了哪些实体类型？\n如果不存在则回答：无\n按照元组形式回复，如 (实体类型1, 实体类型2, ……)：
    prompt = ner_prompt.format(text=text, label=label)
    final_prompt = build_model_prompt(prompt)
    return final_prompt


def RE_attribute_template(text, label, examples=None):
    re_prompt = """抽取语句中以下人物属性：{label}
要求以三元组的形式回答：[人物, 属性类型, 属性]
input：
”发送人: 王涛, 接受人: 张青，主题: 周末出行计划，内容：小青, 你好！最近天气不错, 我们可以趁周末一起出去玩玩。我知道你喜欢摇滚音乐, 我们可以去附近的音乐节现场感受一下摇滚的魅力。我还是比较喜欢宫保鸡丁这道菜”
output：
[王涛, 音乐偏好, 摇滚乐], [张青,  昵称, 小青],  [张青,  菜品偏好, 宫保鸡丁]
        
input:
"{text}"
output:
    """
    # re_prompt = """对于下面语句：{text}，
    #     请抽取语句中包含的以下人物属性类型：{label}，
    #     要求回答以三元组的形式：
    #     [人物，属性类型，属性]
    #     仅输出句子中包含的人物属性，不要输出额外内容。
    # """
    prompt = re_prompt.format(text=text, label=label)
    if not use_local_model:
        prompt = build_model_prompt(prompt)
    return prompt


def RE_relation_template(text, label, examples=None):
    re_prompt = """
你的任务是抽取"输入"中包含的头实体，关系和尾实体，关系必须出现在下面的列表中：{label}，
要求回答以三元组的形式：
输出: [头实体,关系,尾实体]

注意：
1. 不要输出额外内容
2. 请不要输出你假定的内容
3. 请用中文回答。
4. 不要输出示例中的内容

示例:
{example}

任务开始！请不要输出示例中的信息，对下面给到的输入中的人物关系进行抽取。
输入: {text}; 输出:
    """
# 请按照"[头实体,关系,尾实体]"的形式输出
    re_example_prompt = '{number}. 输入: {text}; 输出: {gt}'
    if len(examples) > 15:
        import random
        random.shuffle(examples)
        examples = examples[:15]
    example_prompt = '\n'.join([
        re_example_prompt.format(number=i+1, text=example[0].replace('\n', ''), gt=example[1])
        for i, example in enumerate(examples)
    ])
    # print(example_prompt)
    prompt = re_prompt.format(text=text, label=label, example=example_prompt)
    if not use_local_model:
        prompt = build_model_prompt(prompt)
    print(prompt)
    return prompt


def RE_relation_stage_1(text, label, examples=None):
    re_relation_stage_1_prompt = """
你的任务是给出"输入"中包含的人物关系类型，关系类型必须出现在下面的列表中：{label}，
要求输出形式：
输出: [关系]

注意：
1. 请用中文回答
2. 不要输出额外内容
3. 其中不同关系类型以','分开
4. 如果不存在人物关系或你不确定就返回"无"。

示例:
{example}

任务开始！请不要输出示例中的信息，对下面给到的输入进行抽取。
输入: {text}; 输出: 
"""
    re_relation_example_prompt = '{number}. 输入: {text}; 输出: {gt}'
    if len(examples) > 15:
        import random
        random.shuffle(examples)
        examples = examples[:15]
    example_prompt = '\n'.join([
        re_relation_example_prompt.format(number=i+1, text=example[0].replace('\n', ''), gt=example[1])
        for i, example in enumerate(examples)
    ])

    if use_DuIE:
        prompt = re_relation_stage_1_prompt.format(
            text=text, label=config.DuIE_relation_schema, example=example_prompt)
    else:
        prompt = re_relation_stage_1_prompt.format(
            text=text, label=config.RE_relation_schema_level_1, example=example_prompt)

    # print(prompt)
    if not use_local_model:
        final_prompt = build_model_prompt(prompt)
    else:
        final_prompt = prompt
    # results = requests.request('POST', 'http://localhost:8011/generate',
    #                            json={"prompt": final_prompt,
    #                                  "temperature": 0.1, "max_tokens": 32, "top_k": 10, "top_p": 0.95, "stop": '\n\n'})
    # results = json.loads(results.text)['text'][0]
    # results = results.split('<|assistant|>')[-1]
    results = call_model(final_prompt, 0.1, 32, 10, 0.95, '\n\n')
    print(results)
    words = len(results)

    temp_schema = []
    category_list = []

    if use_DuIE:
        schema_list = copy.deepcopy(config.DuIE_relation_schema)
    else:
        schema_list = copy.deepcopy(config.RE_relation_schema_level_1)

    while True:
        category_pattern = '|'.join(schema_list)
        category = re.search(category_pattern, results)
        if not category:
            break
        category = category.group()
        if category not in category_list:
            category_list.append(category)
        if category in schema_list:
            schema_list.remove(category)
            if not schema_list:
                break

    if use_DuIE:
        return category_list, words

    if not category_list:
        return [], words
    for cate in category_list:
        # print(cate)
        for l in label:
            if config.RE_relation_schema_dict[l].find(cate) != -1:
                temp_schema.append(l)
    # print(temp_schema)
    return temp_schema, words


def EE_stage_1_template(text, label):
    ee_prompt = """事件类型列表：{label}
        给定一句话：{text},
        请判断这句话对应的事件类型，只能从上述列表中选一个，要求回答形式：[]。如果没有事件或你不确定就返回"无"。
    """
    prompt = ee_prompt.format(text=text, label=label)
    if not use_local_model:
        prompt = build_model_prompt(prompt)
    return prompt


def EE_stage_1_template_add_example(text, label, examples=None):
#     ee_prompt = """在如下事件类型列表中：{label}
# 请判断给定的语句中所包含的事件类型，如果没有句子中不包含事件或你不确定就返回["无"]，如果包含事件只能从上述列表中选择一个。
# 要求回答形式：[事件类型]。至少输出一种事件类型或输出"无"。请不要输出额外内容。
# Examples:
# {example}
# 任务开始！
# input: {text} output:
#     """
    ee_prompt = """在如下事件类型列表中：{label}
你的任务是判断给定的语句中所包含的事件类型，如果没有句子中不包含事件或你不确定就返回["无"]，如果包含事件只能从上述列表中选择一个。
要求回答形式：[事件类型]。

注意：
1. 至少输出一种事件类型或输出"无"。
2. 请不要输出额外内容。

Examples:
{example}

任务开始！请不要输出示例中的信息，对下面给到的输入进行抽取。
input: {text}; output:
    """
    ee_example_prompt = '{number}. input: {text}; output: {gt}'
    example_prompt = '\n'.join([
        ee_example_prompt.format(number=i+1, text=example[0], gt=example[1]) for i, example in enumerate(examples)])
    print(example_prompt)
    prompt = ee_prompt.format(text=text, label=label, example=example_prompt)
    if not use_local_model:
        prompt = build_model_prompt(prompt)
    return prompt


def EE_stage_2_template(text, label):
    ee_prompt = """事件类型：{label}
        请判断给定的语句中对应的唯一事件类型，只能从上述列表中选一个，要求回答形式：[]
        input:
        发送人: 王阳, 接受人: 刘雨，主题: 明天的约定，内容：刘雨, 明天我们一起去你的家乡杭州市旅游吧, 我已经帮你预订好了酒店, 地址是浙江省杭州市, 希望你能喜欢。
        output:
        ['旅游']
        input:
        {text}
        output:
    """
    prompt = ee_prompt.format(text=text, label=label)
    if not use_local_model:
        prompt = build_model_prompt(prompt)
    return prompt


def EE_stage_2_template_add_example(text, label, examples=None):
    # ee_prompt = """请输出input中包含的事件类型，事件类型必须出现在下面的列表中：{label}，
    # 要求回答形式：
    # output: []
    # Examples：
    # {example}
    # 任务开始！
    # input: {text} output:
    # """
    ee_prompt = """你的任务是输出input中包含的事件类型，事件类型必须出现在下面的列表中：{label}，
要求回答形式：
output: []

注意：
1. 请不要输出额外的内容

Examples：
{example}

任务开始！请不要输出示例中的信息，对下面给到的输入进行抽取。
input: {text}; output:
"""
    ee_example_prompt = '{number}. input: {text}; output: {gt}'
    example_prompt = '\n'.join([
        ee_example_prompt.format(number=i+1, text=example[0], gt=example[1]) for i, example in enumerate(examples)])
    prompt = ee_prompt.format(text=text, label=label, example=example_prompt)
    print(example_prompt)
    if not use_local_model:
        prompt = build_model_prompt(prompt)
    return prompt


def EE_stage_3_template(text, label, category):
    ee_prompt = """对于{category}事件类型，
        对于下面语句：{text}，
        请分别抽取以下要事件要素：{label}，
        要求回答形式：
        [要素]:[ ](其中的内容如人名请以";"分隔，无内容或未提供请回答"<无>")
        输出的时候请不要保留[]
        如果某个事件要素没有相应的内容，请回答"<无>"，请仅用中文回答。
    """
    prompt = ee_prompt.format(text=text, label=label, category=category)
    if not use_local_model:
        prompt = build_model_prompt(prompt)
    return prompt


def task_planning(text, label, examples, task):
    plan_prompt = """你的任务是判断给定的句子中是否包含{task}类型。

    在如下{task}类型列表中：{label}。

    如果句子中不包含相关{task}或你无法确定就返回['无']。
    如果包含{task}类型只能从伤处列表中选择

    注意：
    1.至少输出一种{task}类型或输出‘无’
    2.请不要输出句子中不存在的内容
    3.请不要输出额外内容

    要求回答形式：[]

    参考示例：
    {examples}

    任务开始！
    输入：{text}; 输出：
    """
    tp_example_prompt = '{number}. 输入: {text}; 输出: {gt}'
    example_prompt = '\n'.join([
        tp_example_prompt.format(number=i+1, text=example[0], gt=example[1]) for i, example in enumerate(examples)])
    prompt = plan_prompt.format(text=text, label=label, examples=example_prompt, task=task)
    if not use_local_model:
        prompt = build_model_prompt(prompt, role='你是一个优秀的个人数据分析专家，你需要负责有逻辑地组织必要的抽取任务用于挖掘个人信息。')
    return prompt


def optimize_stage_2_template(text, label, element):
    op_2_prompt = """你的任务是判断给定的句子中是否包含与{label}相关的{element}.
      
      注意：
      1.请不要输出额外内容
      2.请只关注句子中存在的内容，不要进行不必要的推理和想象
      3.如果句子中不存在相关信息，请直接输出‘无’

      任务开始！
      输入：{text}
      输出：
    """
    prompt = op_2_prompt.format(text=text, label=label, element=element)
    if not use_local_model:
        prompt = build_model_prompt(prompt, role='你是一个专业且严谨的文本理解专家，你会根据文本中呈现的内容进行准确的判断。')
    return prompt


def optimize_stage_3_template(text, label, element, task):
    op_3_prompt = """你的任务是判断下面句子中{label}{task}的{element}是什么？

    注意:
    1.从句子中能够判断，请以“{element}:”的形式输出
    2.从句子中无法推断具体的{element}.请直接输出‘无’
    3.不要输出额外的内容

    任务开始！
    输入：{text}.输出：
    """
    prompt = op_3_prompt.format(text=text, label=label, element=element, task=task)
    if not use_local_model:
        prompt = build_model_prompt(prompt, role='你是一个专业且严谨的文本理解专家，你会根据文本中呈现的内容进行准确的判断。')
    return prompt


def NER(text, schema=None):
    result = utils.get_empty_result(ner=True)

    schema = schema if schema else config.NER_schema
    prompt = NER_template(text, schema)
    # results = pipeline_ins(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
    # results = results[0]['generated_text']
    # results = requests.request('POST', 'http://localhost:8011/generate',
    #                            json={"prompt": prompt,
    #                                  "temperature": 0.1, "max_tokens": 128, "top_k": 10, "top_p": 0.95, "stop": '\n\n'})
    # # print(results.text)
    # results = json.loads(results.text)['text'][0]
    # results = results.split('<|assistant|>')[-1]
    results = call_model(prompt, 0.1, 128, 10, 0.95, '\n\n')
    print(colorama.Fore.BLUE + results)
    print('---')
    results = results.split('\n\n')[0]
    results = results.replace(']', ']\n')
    results = results.split('\n')
    for line in results:
        line = utils.fix_str(line)
        if line.find(':') == -1:
            continue
        entity_type, entities = line.split(':')[0], ':'.join(line.split(':')[1:])
        for s in schema:
            if entity_type.find(s) != -1:
                entity_type = s
                break
        if entity_type not in schema:
            continue
        entities = entities.split(';')
        for entity in entities:
            entity = utils.fix_str(entity)
            if entity not in ['无', '未提供']:
                data = utils.transfer_ner_result(entity, utils.fix_str(entity_type))
                if data not in result['named_entity']:
                    result['named_entity'].append(data)

    print(result)
    print('---')
    return result


def RE(text, schema=None, relation_type='attribute'):
    if not use_DuIE:
        text = text.replace('\n', '')
    l1_dict, l2_dict = utils.get_re_example_dict()
    duie_dict, _ = utils.get_DuIE_example_dict()
    if text.find('发送人') == -1:
        text_type = 'note'
    elif text.find('主题') == -1:
        text_type = 'message'
    else:
        text_type = 'email'
    # else:
    #     text_type = 'message'

    # print(l1_dict[text_type], l2_dict[text_type])

    result = utils.get_empty_result(re=True)
    extra_result = utils.get_empty_result(re=True)

    if relation_type == 'attribute':
        schema = config.RE_attribute_schema if not schema else schema
        schema_dict = config.RE_attribute_schema_dict
        prompt_template = RE_attribute_template
        stage_1 = lambda x, y, z: (y, 0)
        examples = None
    else:
        if not use_DuIE:
            schema = config.RE_relation_schema if not schema else schema
            schema_dict = config.RE_relation_schema_dict
            prompt_template = RE_relation_template
            stage_1 = RE_relation_stage_1
            examples = utils.get_re_example(l1_dict[text_type], config.RE_relation_schema_level_1, 2)
        else:
            schema = config.DuIE_relation_schema
            schema_dict = {}
            for s in schema:
                schema_dict[s] = s
            prompt_template = RE_relation_template
            stage_1 = lambda x, y, z: (y, 0)
            examples = utils.get_re_example(duie_dict, schema, 1)

    # print(examples)
    schema, words_1 = stage_1(text, schema, examples)
    print(schema)
    if not schema:
        return result, words_1
    num = 3 if len(schema) < 15 else 1
    # num = 1

    if use_DuIE:
        examples_2 = utils.get_re_example(duie_dict, schema, num)
    else:
        examples_2 = utils.get_re_example(l2_dict[text_type], schema, num)

    # print(examples_2)
    prompt = prompt_template(text, schema, examples_2)
    # print(prompt)
    # results = requests.request('POST', 'http://localhost:8011/generate',
    #                            json={"prompt": prompt,
    #                                  "temperature": 0.1, "max_tokens": 64, "top_k": 10, "top_p": 0.95, "stop": '\n\n'})
    # # print(results.text)
    # results = json.loads(results.text)['text'][0]
    # results = results.split('<|assistant|>')[-1]
    results = call_model(prompt, 0.1, 64, 10, 0.95, '\n\n')
    print(colorama.Fore.YELLOW + results)
    print('---')
    words_2 = len(results)

    pattern = re.compile('\[[^\[\]]*,[^\[\]]*,[^\[\]]*\]')
    # results = results.replace('，', ',')
    relations = pattern.findall(results)

    escape_list = ['未知', '自身', '未提到的人', '语句中提到的对方', '语句中提到的自我描述', '语句中未提到的人',
                   '未提供', '自己', '输入者', '收信人', '发件人', 'N/A', '12306', '美团', '发信人', '未提', '分隔']
    escape_chr = ['-', '?', '无', '12306', '赵梓', '杨涛']
    for relation in relations:
        relation = relation[1:-1]
        extract_relation = relation.split(',')
        head, key, tail = extract_relation[0], extract_relation[1], ','.join(extract_relation[2:])
        head, tail = head.split('(')[0], tail.split('(')[0]
        head, key, tail = utils.fix_str(head), utils.fix_str(key), utils.fix_str(tail)
        if head in escape_list or tail in escape_list \
                or head in escape_chr or tail in escape_chr \
                or not head or not tail \
                or head == tail \
                or sum([head.find(escape) for escape in escape_list]) != -len(escape_list) \
                or sum([tail.find(escape) for escape in escape_list]) != -len(escape_list) \
                or not utils.check_relation_value(head, key, tail):
            continue

        if key == '工作地址':
            key = '公司地址'

        if key in schema:
            tail_type = '人名' if relation_type == 'relation' else config.RE_attribute_type_schema_dict[
                schema_dict[key]]
            data = utils.transfer_re_result(head, schema_dict[key], tail, head_type='人名', tail_type=tail_type)
            if data not in result['relation']:
                result['relation'].append(data)
        else:
            extra_result['relation'].append(utils.transfer_re_result(head, key, tail))

    if extra_result['relation']:
        print('extra results:', extra_result)
        with open(config.out_schema_result_path, 'a', encoding='utf-8') as f:
            f.write(str({
                'text': text,
                'results': extra_result
            }) + '\n')

    print(result)
    return result, words_1 + words_2


def EE(text, schema=None, flag=True):
    l1_dict, l2_dict, email_l1_dict, email_l2_dict, note_l1_dict, note_l2_dict = utils.get_ee_example_dict()
    duee_l1_dict, duee_l2_dict, _ = utils.get_DuEE_example_dict()
    result = utils.get_empty_result(ee=True)
    extra_result = utils.get_empty_result(ee=True)
    if text.find('主题') != -1:
        l1, l2 = email_l1_dict, email_l2_dict
    elif text.find('发送人') == -1:
        l1, l2 = note_l1_dict, note_l2_dict
    else:
        l1, l2 = l1_dict, l2_dict

    if use_DuEE:
        schema = config.EE_DuEE_schema_dict
        schema_level = config.EE_DuEE_schema_level_dict
    else:
        schema = schema if schema else config.EE_schema_dict
        schema_level = config.EE_schema_level_dict

    words_1, words_2, words_3 = 0, 0, 0

    if flag:
        # stage_1_prompt = EE_stage_1_template(text, list(schema_level.keys()))
        if use_DuEE:
            stage_1_prompt = EE_stage_1_template_add_example(
                text, list(schema_level.keys()), utils.get_ee_stage1_example(duee_l1_dict, 2))
        else:
            stage_1_prompt = EE_stage_1_template_add_example(
                text, list(schema_level.keys()), utils.get_ee_stage1_example(l1, 1))
        # results = requests.request('POST', 'http://localhost:8011/generate',
        #                            json={"prompt": stage_1_prompt,
        #                                  "temperature": 0.1, "max_tokens": 32, "top_k": 10, "top_p": 0.95, "stop": '\n\n'})
        # print(colorama.Fore.GREEN + results.text)
        # results = json.loads(results.text)['text'][0]
        # results = results.split('<|assistant|>')[-1]
        results = call_model(stage_1_prompt, 0.1, 32, 10, 0.95, '\n\n')
        words_1 = len(results)
        category_1_pattern = '|'.join(list(schema_level.keys()))
        category_1 = re.search(category_1_pattern, results)
        if not category_1 or results.find('无') != -1:
            return result, words_1

        category_1 = category_1.group(0)
        # stage_2_prompt = EE_stage_2_template(text, schema_level[category_1])
        if use_DuEE:
            stage_2_prompt = EE_stage_2_template_add_example(
                text, schema_level[category_1], utils.get_ee_stage2_example(duee_l2_dict, schema_level[category_1], 2))
        else:
            stage_2_prompt = EE_stage_2_template_add_example(
                text, schema_level[category_1], utils.get_ee_stage2_example(l2, schema_level[category_1], 1))
        # results = requests.request('POST', 'http://localhost:8011/generate',
        #                            json={"prompt": stage_2_prompt,
        #                                  "temperature": 0.1, "max_tokens": 32, "top_k": 10, "top_p": 0.95, "stop": '\n\n'})
        # print(colorama.Fore.GREEN + results.text)
        # results = json.loads(results.text)['text'][0]
        # results = results.split('<|assistant|>')[-1]
        results = call_model(stage_2_prompt, 0.1, 32, 10, 0.95, '\n\n')
        words_2 = len(results)
        category_2_pattern = '|'.join(schema_level[category_1])
        category_2 = re.search(category_2_pattern, results)
        if category_2:
            category_2 = category_2.group(0)
            category = category_1 + '/' + category_2 if not use_DuEE else category_1 + '-' + category_2
        else:
            category = '其他' 
    else:
        category = '其他' 

    # labels = schema[category] + ['事件描述']
    labels = schema[category]
    if '参与者' not in labels and not use_DuEE:
        labels += ['参与者']
    # if '时间' not in labels:
    #     labels = ['时间'] + labels

    stage_3_prompt = EE_stage_3_template(text, labels, category)
    # results = requests.request('POST', 'http://localhost:8011/generate',
    #                            json={"prompt": stage_3_prompt,
    #                                  "temperature": 0.1, "max_tokens": 128, "top_k": 10, "top_p": 0.95, "stop": '\n\n'})
    # # print(results.text)
    # results = json.loads(results.text)['text'][0]
    # results = results.split('<|assistant|>')[-1]
    results = call_model(stage_3_prompt, 0.1, 128, 10, 0.95, '\n\n')
    words_3 = len(results)
    results = results.split('\n\n例如')[0]
    results = results.replace('[', '')
    results = results.replace(']', '')
    results = results.replace('。', '')
    print(results)
    roles = []
    for label in labels:
        if results.find(label) != -1 and results.find('\n' + label) == -1:
            results = results.replace(label, '\n' + label)
    for label in labels:
        answer = re.findall(r'{label}[:：](.*)'.format(label=label), results)
        if not answer:
            continue
        answer = answer[0]

        if answer.find('<无>') != -1 or answer.find('未提供') != -1 \
                or answer == '-':
            continue

        answer = utils.fix_str(answer)
        answer = answer.replace('请以";"分隔', '')
        if answer.find(',') <= answer.find('(') or answer.find('，') <= answer.find('('):
            answer = answer.replace(',', ';')
            answer = answer.replace('，', ';')
        answer = answer.replace('、', ';')
        answer = answer.replace('和', ';')
        # print(answer)
        if label != '事件描述':
            value = []
            for ans in answer.split(';'):
                v = utils.fix_str(ans.split('(')[0])
                v = v.replace('收信人', '')
                v = v.replace('发件人', '')
                if v:
                    if label == '参与者':
                        v = v.split(':')[-1]
                    value.append({
                        'text': v
                    })
            roles.append({
                'key': label,
                'value': value
            })
        else:
            roles.append({
                'key': label,
                'value': [{'text': utils.fix_str(answer)}]
            })

    optimize_labels = utils.check_optimize(roles, labels)
    for label in optimize_labels:
        pass

    final_key, final_value = utils.transfer_ee_result({
        'key': category + '触发词',
        'text': None
    }, roles)
    if final_value['relations']:
        result['event'][final_key] = [final_value]

    print(result)
    return result, words_1 + words_2 + words_3


def check_EE_stage1(text, schema=None):
    l1_dict, l2_dict, note_l1_dict, note_l2_dict = utils.get_ee_example_dict()
    result = utils.get_empty_result(ee=True)

    schema = schema if schema else config.EE_schema_dict
    schema_level = config.EE_schema_level_dict

    stage_1_prompt = EE_stage_1_template_add_example(
        text, list(schema_level.keys()), utils.get_ee_stage1_example(l1_dict, 1))
    # stage_1_prompt = EE_stage_1_template(
    #     text, list(schema_level.keys()))
    results = requests.request('POST', 'http://localhost:8011/generate',
                               json={"prompt": stage_1_prompt,
                                     "temperature": 0.1, "max_tokens": 32, "top_k": 10, "top_p": 0.95,
                                     "stop": '\n\n'})
    # print(results.text)
    results = json.loads(results.text)['text'][0]
    results = results.split('<|assistant|>')[-1]
    # print(results)
    category_1_pattern = '|'.join(list(schema_level.keys()))
    category_1 = re.search(category_1_pattern, results)
    if not category_1 or results.find('无') != -1:
        return '无'
    category_1 = category_1.group(0)

    # stage_2_prompt = EE_stage_2_template(text, schema_level[category_1])
    stage_2_prompt = EE_stage_2_template_add_example(
        text, schema_level[category_1], utils.get_ee_stage2_example(l2_dict, schema_level[category_1], 1))
    results = requests.request('POST', 'http://localhost:8011/generate',
                               json={"prompt": stage_2_prompt,
                                     "temperature": 0.1, "max_tokens": 32, "top_k": 10, "top_p": 0.95, "stop": '\n\n'})
    # print(colorama.Fore.GREEN + results.text)
    results = json.loads(results.text)['text'][0]
    results = results.split('<|assistant|>')[-1]
    category_2_pattern = '|'.join(schema_level[category_1])
    category_2 = re.search(category_2_pattern, results)
    if category_2:
        category_2 = category_2.group(0)
        category = category_1 + '/' + category_2
    else:
        category = '其他'
    return category


def IE(text, schema=None):
    if schema is None:
        schema = [None, None, None, None]
    result = utils.get_empty_result(ie=True)

    retry = 1

    # result_entity = NER(text, schema[0])
    for i in range(retry):
        result_attribute, words_attr = RE(text, schema[1], 'attribute')
        if result_attribute['relation']:
            break
    for i in range(retry):
        result_relation, words_rel = RE(text, schema[2], 'relation')
        if result_relation['relation']:
            break
    # for i in range(retry):
    #     result_event, words_event = EE(text, schema[3])
    #     if result_event['event']:
    #         break
    # if not result_event['event']:
    #     result_event, words_event = EE(text, schema[3], False)

    # result['named_entity'] = result_entity['named_entity']
    result['relation'] = result_attribute['relation'] + result_relation['relation']
    # result['relation'] = result_relation['relation']
    # result['event'] = result_event['event']
    # result['words'] = words_attr + words_rel + words_event

    print(colorama.Fore.LIGHTWHITE_EX + str(result))
    return result


if __name__ == '__main__':
    # import time
    # start = time.time()
    # build_model_prompt('你好')
    # NER('大家好, 明天是公司年度体检, 记得带上1000元费用。张伟、王明和张三都会参加。 ')
    # NER('大家好, 明天是公司年度体检, 记得带上1000元费用。张伟、王明和张三都会参加。 ')
    # NER('大家好, 明天是公司年度体检, 记得带上1000元费用。张伟、王明和张三都会参加。 ')
    # NER('10月1号我们去京福华吃肥牛火锅吧')
    # NER('10月1号我们去京福华吃肥牛火锅吧')
    # NER('10月1号我们去京福华吃肥牛火锅吧')
    # NER('小明, 你好！我发现你的身高是175cm, 体重是70kg, 身材真棒！希望我们可以一起健身, 互相激励, 成为更好的自己。')
    # RE('小明, 你好！我发现你的身高是175cm, 体重是70kg, 身材真棒！希望我们可以一起健身, 互相激励, 成为更好的自己。', relation_type='attribute')
    RE('创建人：杨梓,内容：刘宇是我的小学同学', relation_type='relation')
    # NER('小明, 你好！我发现你的身高是175cm, 体重是70kg, 身材真棒！希望我们可以一起健身, 互相激励, 成为更好的自己。')
    # EE('消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了')
    # IE('内容: 大家好, 明天是公司年度体检, 记得带上1000元费用。张伟、王明和张三都会参加。 发送人: 杨梓, 接受人: 公司同事')
    # IE('您好!明天上午，大会设置了合影环节，如有意向参加，您可于08:00乘坐希尔顿酒店门口的摆渡车前往中财大厦大堂门前集合点。如自行前往，请您务必于08:30到合影地点集合。')
    # RE('')
    # RE('嘿刘洋,我是杨梓!好久不见!怀念我们大学时代的美好时光,还记得我们一起参加社团活动、一起熬夜复习的日子吗? 真的很想念你,希望我们能常常保持联系!', relation_type='relation')
    # print(check_EE_stage1('发送人: 杨梓, 接受人: 王晶，主题: 关于过去的合作，内容：王晶, 你好！希望一切都好。想起我们在公司共事的日子, 感觉时光飞逝。不知道你还记得吗？我们当时是同事关系, 一起努力工作, 相互配合。记得那次我们一起完成的项目, 真是辛苦但也很有成就感。重温过去的点滴, 也希望我们能有机会再次合作, 共同创造更多的成功。期待你的回复！'))
    # print(time.time() - start)

    # RE('《东华春理发厅》改编自阮光民的同名漫画，是由蔡东文执导，阮光民编剧，六月、刘至翰、曾少宗、杨雅筑、李至正等主演的时装言情剧', relation_type='relation')
    # RE('《淡忘如思，回眸依旧》作者：小懒内容简介：顾箐如曾说，沈思彦 ，因为你，我想看到这个世界', relation_type='relation')
    # RE('令狐静，字修齐，天武皇帝刘仪宠妃，成刚小说《最大帝》女主', relation_type='relation')
    # RE('发送人: 杨梓, 接收人: 赵晓, 内容: 嘿！赵晓, 这里是杨梓。好久不见了, 最近怎么样？我们大学的回忆还记得吗？', relation_type='relation')
    # RE('发送人: 杨梓, 接收人: 刘洋, 内容: 嘿刘洋, 我是杨梓！好久不见！怀念我们大学时代的美好时光, 还记得我们一起参加社团活动、一起熬夜复习的日子吗？真的很想念你, 希望我们能常常保持联系！', relation_type='relation')
    # RE('发件人: 杨梓, 收件人: 李青, 主题: 大学同学的问候, 内容: 亲爱的李青, 好久不见！作为大学同学, 我们共同经历了许多难忘的时光, 无论是一起上课、一起做项目, 还是一起度过无数个夜晚的熬夜复习, 这些回忆都成为我宝贵的财富。祝你一切顺利！杨梓', relation_type='relation')
    # RE('发送人: 杨梓, 接收人: 李晴, 内容: 亲爱的李晴, 昨天和你聊天真的很开心, 感觉我们有很多共同话题。很高兴成为你的朋友，期待我们下一次一起出去玩！', relation_type='relation')
    # EE('发件人: 李梅, 收件人: 杨梓, 主题: 关于海尔集团客户商务洽谈会的通知, 内容: 尊敬的杨梓, 本人与您共同参加海尔集团客户商务洽谈会, 会议将于2023-11-29下午三点在杭州希尔顿酒店举行。请您准时参加, 并做好相关准备工作。参与者包括杨梓、李梅、刘晓、陈杨、刘红、王丹、杨冰、赵丽。谢谢。')
    # RE('发送人: 杨梓, 接收人: 刘晓霞, 内容: 亲爱的妈妈, 母亲节快乐！我希望您身体健康、心情愉快。首先, 我希望您一切都好。这段时间我过得很充实, 我在学校认真学习, 努力进步。母亲, 您是我生命中最重要的人, 是我永远的依靠和支持。无论我有什么困难或烦恼, 您都会耐心倾听并给予我最中肯的建议。祝愿您幸福安康, 心想事成。', relation_type='relation')
    exit()
    from tqdm import tqdm

    file = open('../label_11_7.json', 'r', encoding='utf-8')
    out = open('./l1-output.txt', 'w')
    data = json.load(file)
    total = 0
    empty, correct_empty = 0, 0
    l1_correct = 0
    l2_correct = 0
    for d in tqdm(data[:156]):
        gt = d['result']['event']
        text = d['text']
        output = check_EE_stage1(text)

        if not gt:
            gt_keys = '无'
            empty += 1
            if output == '无':
                correct_empty += 1
        else:
            gt_keys = list(gt.keys())[0].split('触发词')[0]
            if gt_keys.split('/')[0] == output.split('/')[0]:
                l1_correct += 1
            if '/'.join(gt_keys.split('/')[1:]) == '/'.join(output.split('/')[1:]):
                l2_correct += 1

        out.write(output + ' ' + gt_keys + '\n')
        # out.write(output + ' ' + gt_keys + ' ' + text + '\n')
        out.flush()

    print('l1 correct', l1_correct, len(data) - empty)
    print('l1 l2 correct', l2_correct, len(data) - empty)
    print('empty correct', correct_empty, empty)
