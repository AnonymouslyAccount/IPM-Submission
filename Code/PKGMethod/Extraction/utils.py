def get_empty_result(ner=False, re=False, ee=False, ie=False):
    result = {}

    if ner or ie:
        result.setdefault('named_entity', [])
    if re or ie:
        result.setdefault('relation', [])
    if ee or ie:
        result.setdefault('event', {})

    return result


def transfer_ner_result(entity, entity_type, entity_start_index=-1, score=-1):
    return {
        'entity': entity,
        'entity_type': entity_type,
        'entity_start_index': entity_start_index,
        'score': score
    }


def transfer_re_result(head, relation, tail, score=-1,
                       head_type='None', head_start_index=-1,
                       tail_type='None', tail_start_index=-1):
    return {
        'relation_type': relation,
        'head': head,
        'head_entity_type': head_type,
        'head_start_index': head_start_index,
        'tail': tail,
        'tail_entity_type': tail_type,
        'tail_start_index': tail_start_index,
        'score': score
    }


def transfer_ee_result(trigger, roles):
    key = trigger['key']

    value = {
        'text': trigger['text'],
        'start': trigger.get('start', -1),
        'end': trigger.get('end', -1),
        'probability': trigger.get('probability', -1),
        'relations': {}
    }

    for role in roles:
        value['relations'][role['key']] = [{
            'text': role_value['text'],
            'start': role_value.get('start', -1),
            'end': role_value.get('end', -1),
            'probability': role_value.get('probability', -1)
        } for role_value in role['value']]

    return key, value


def check_relation_value(head, key, tail):
    import re

    if key in ['身份证号', '银行卡号', '电话号码']:
        if not re.search('^[0-9*]*$', tail):
            return False

    if key in ['婚姻情况']:
        if tail not in ['已婚', '未婚']:
            return False

    return True


def fix_str(s):
    s = s.strip()
    s = s.removeprefix('\'')
    s = s.removeprefix('\"')
    s = s.removeprefix('<')
    s = s.removeprefix('[')
    s = s.removeprefix('(')
    s = s.removeprefix('（')
    s = s.removesuffix('\'')
    s = s.removesuffix('\"')
    s = s.removesuffix('>')
    s = s.removesuffix(']')
    s = s.removesuffix(')')
    s = s.removesuffix('）')
    s = s.removesuffix('！')
    s = s.removesuffix('!')
    s = s.replace('：', ':')
    s = s.replace('；', ';')
    return s


def get_ee_example_dict():
    import json
    l1_dict, l2_dict = {}, {}
    email_l1_dict, email_l2_dict = {}, {}
    note_l1_dict, note_l2_dict = {}, {}
    file_list = ['../label_11_7.json', '../note.json', '../note_attribute.json']

    for file in file_list:
        f = open(file, 'r')
        data = json.load(f)
        for d in data:
            res = d['result']
            if not res['event']:
                if file.find('note') != -1:
                    if '无' in note_l1_dict:
                        note_l1_dict['无'].append(d['text'])
                    else:
                        note_l1_dict['无'] = [d['text']]
                elif d['text'].find('主题') == -1:
                    if '无' in l1_dict:
                        l1_dict['无'].append(d['text'])
                    else:
                        l1_dict['无'] = [d['text']]
                else:
                    if '无' in email_l1_dict:
                        email_l1_dict['无'].append(d['text'])
                    else:
                        email_l1_dict['无'] = [d['text']]
            else:
                event_type = list(res['event'].keys())[0].split('触发词')[0]
                event_type, l2_type = event_type.split('/')[0], '/'.join(event_type.split('/')[1:])
                if file.find('note') != -1:
                    if event_type in note_l1_dict:
                        note_l1_dict[event_type].append(d['text'])
                    else:
                        note_l1_dict[event_type] = [d['text']]
                    if l2_type in note_l2_dict:
                        note_l2_dict[l2_type].append(d['text'])
                    else:
                        note_l2_dict[l2_type] = [d['text']]
                elif d['text'].find('主题') == -1:
                    if event_type in l1_dict:
                        l1_dict[event_type].append(d['text'])
                    else:
                        l1_dict[event_type] = [d['text']]
                    if l2_type in l2_dict:
                        l2_dict[l2_type].append(d['text'])
                    else:
                        l2_dict[l2_type] = [d['text']]
                else:
                    if event_type in email_l1_dict:
                        email_l1_dict[event_type].append(d['text'])
                    else:
                        email_l1_dict[event_type] = [d['text']]
                    if l2_type in email_l2_dict:
                        email_l2_dict[l2_type].append(d['text'])
                    else:
                        email_l2_dict[l2_type] = [d['text']]
    return l1_dict, l2_dict, email_l1_dict, email_l2_dict, note_l1_dict, note_l2_dict


def get_ee_stage1_example(text_dict, num=1):
    import random
    random.seed = 1234
    examples = []
    for key in text_dict.keys():
        if key in ['美食', '生活', '休闲娱乐']:
            n = 3
        else:
            n = num
        if len(text_dict[key]) < n:
            ex = text_dict[key]
        else:
            ex = random.sample(text_dict[key], n)
        for e in ex:
            examples.append([e, '["' + key + '"]'])
    # print(examples)
    return examples


def get_ee_stage2_example(text_dict, keys, n=1):
    import random
    random.seed = 1234
    examples = []
    for key in keys:
        if key not in text_dict:
            continue
        if len(text_dict[key]) < n:
            ex = text_dict[key]
        else:
            ex = random.sample(text_dict[key], n)
        for e in ex:
            examples.append([e, '["' + key + '"]'])
    return examples


def get_DuEE_example_dict():
    import json
    l1_dict, l2_dict, hack_dict = {}, {}, {}
    file = open('../ee_data.json', 'r', encoding='utf-8')
    data = json.load(file)

    for d in data:
        text = d['text']
        for category in d['event']:
            l1, l2 = category[:-3].split('-')
            if l1 not in l1_dict:
                l1_dict[l1] = [text]
            else:
                l1_dict[l1].append(text)
            if l2 not in l2_dict:
                l2_dict[l2] = [text]
            else:
                l2_dict[l2].append(text)
        hack_dict[text] = list(d['event'].keys())[0][:-3]
    return l1_dict, l2_dict, hack_dict


def get_re_example_dict():
    import json
    l1_dict = {
        'message': {},
        'email': {},
        'note': {}
    }
    l2_dict = {
        'message': {},
        'email': {},
        'note': {}
    }
    # import os
    # print(os.getcwd())
    file_list = ['../label_11_7.json', '../note.json', '../note_attribute.json']

    for file in file_list:
        f = open(file, 'r', encoding='utf-8')
        all_data = json.load(f)
        for data in all_data:
            relations = data['result']['relation']
            text = data['text']

            if file.find('note') != -1:
                level_1, level_2 = l1_dict['note'], l2_dict['note']
            elif text.find('主题') != -1:
                level_1, level_2 = l1_dict['email'], l2_dict['email']
            else:
                level_1, level_2 = l1_dict['message'], l2_dict['message']

            relation_type_l1_list, relation_type_l2_list, relation_list = [], [], []
            for relation in relations:
                head, key, tail = relation['head'], relation['relation_type'], relation['tail']
                if key.find('-') != -1:
                    l1, l2 = key.split('-')
                else:
                    l1, l2 = key, key
                if l1 not in relation_type_l1_list:
                    relation_type_l1_list.append(l1)
                if l2 not in relation_type_l2_list:
                    relation_type_l2_list.append(l2)
                relation_list.append(str([head, key.split('-')[-1], tail]))
            for relation_type_l1 in relation_type_l1_list:
                if relation_type_l1 in level_1:
                    level_1[relation_type_l1].append([text, str(relation_type_l1_list)])
                else:
                    level_1[relation_type_l1] = [[text, str(relation_type_l1_list)]]
            for relation_type_l2 in relation_type_l2_list:
                if relation_type_l2 in level_2:
                    level_2[relation_type_l2].append([text, ','.join(relation_list)])
                else:
                    level_2[relation_type_l2] = [[text, ','.join(relation_list)]]
    return l1_dict, l2_dict


def get_re_example(text_dict, keys, num=1):
    import random
    # random.seed = 4321
    examples = []
    # print(text_dict.keys())
    # print(keys)
    for key in keys:
        if key not in text_dict:
            continue
        if len(text_dict[key]) < num:
            example = text_dict[key]
        else:
            example = random.sample(text_dict[key], num)
        examples.extend(example)
    return examples


def get_DuIE_example_dict():
    import json
    result_dict = {}
    hack_dict = {}

    file = open('../rel_data.json', 'r', encoding='utf-8')
    all_data = json.load(file)
    for data in all_data:
        relations = data['relation']
        text = data['text']

        relation_type_list, relation_list = [], []
        for relation in relations:
            head, key, tail = relation['head'], relation['relation_type'], relation['tail']
            if key not in relation_type_list:
                relation_type_list.append(key)
            relation_list.append(str([head, key, tail]))
        for relation_type in relation_type_list:
            if relation_type in result_dict:
                result_dict[relation_type].append([text, ','.join(relation_list)])
            else:
                result_dict[relation_type] = [[text, ','.join(relation_list)]]
        hack_dict[text] = relation_type_list
    return result_dict, hack_dict


def check_optimize(roles, labels):
    possible_labels = []
    optimize_labels = []
    for label in labels:
        if label.find('时间') != -1 or label == '参与者' or label == '地点':
            possible_labels.append(label)
    for label in possible_labels:
        flag = True
        for role in roles:
            if role['key'] == label:
                flag = False
                break
        if flag:
            optimize_labels.append(label)
    return optimize_labels


if __name__ == '__main__':
    import config
    a, b, c, d = get_ee_example_dict()
    # print(b.keys())
    # print(a)
    # print(b)
    print(get_ee_stage1_example(c, 3))

    # a, b, c = get_DuEE_example_dict()
    # # print(a)
    # print(a.keys())
    # print(c)
    # print(get_ee_stage2_example(a, config.EE_DuEE_schema_level_dict.keys(), 2))
