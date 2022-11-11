import jsonlines
import torch
from transformers import BatchEncoding

def add_dic(dic1, dic2):
    for key in dic2.keys():
        dic1[key]+=dic2[key]
    return dic1

def append_dic(dic1, dic2):
    for key in dic2.keys():
        dic1[key].append(dic2[key])
    return dic1

# entity-attribute 리스트 갯수만큼
def make_ent_attr_data(sentence, entity_list, tokenizer):
    tokenized_sentence_list = []
    tokenized_attention_mask_list = []
    token_type_index_list = []

    for entity in entity_list:
        temp_list = tokenizer(sentence, entity)
        tokenized_sentence_list.append(temp_list.data['input_ids'])
        tokenized_attention_mask_list.append(temp_list.data['attention_mask'])
        token_type_index_list.append(temp_list.data['token_type_ids'])

    res = {'input_ids': tokenized_sentence_list, 'attention_mask': tokenized_attention_mask_list, 'token_type_ids':token_type_index_list}
    return res

def make_padded_list(tokenized_ent_attr_list, tokenizer):
    max_len = len(max(tokenized_ent_attr_list['input_ids'], key=len))
    res ={'input_ids':[], 'attention_mask':[], 'token_type_ids':[]}
    for input_ids, attention_mask, token_type_ids in zip(tokenized_ent_attr_list['input_ids'], tokenized_ent_attr_list['attention_mask'], tokenized_ent_attr_list['token_type_ids']):
        temp_input_ids = input_ids.copy()
        temp_attention_mask = attention_mask.copy()
        temp_token_type_ids = token_type_ids.copy()
        for i in range(max_len-len(input_ids)):
            temp_input_ids.append(tokenizer.pad_token_id)
            temp_attention_mask.append(0)
            temp_token_type_ids.append(0)
        res['input_ids'].append(temp_input_ids)
        res['attention_mask'].append(temp_attention_mask)
        res['token_type_ids'].append(temp_token_type_ids)
    return res

def jsonl2Diclist(filepath):
    output_list = []
    with jsonlines.open(filepath) as f:
        for line in f:
            output_list.append({'id':line['id'], 'sentence_form':line['sentence_form'],'annotation': line['annotation']})
    return output_list

def diclist2dic(diclist):
    keys = diclist[0].keys()
    temp=[]
    newdic={}
    for key in keys:
        for dic in diclist:
            temp.append(dic[key])
        newdic[key] = temp.copy()
        temp=[]
    return newdic

def label_ent_attr(sentence, annotation_list, entity_list, tokenizer):
    tokenized_ent_attr_list = make_ent_attr_data(sentence, entity_list, tokenizer)
    tokenized_ent_attr_list = make_padded_list(tokenized_ent_attr_list, tokenizer)
    tokenized_ent_attr_list['label'] = [0 for _ in range(len(entity_list))]
    for annotation in annotation_list:
        tokenized_ent_attr_list['label'][entity_list.index(annotation[0])] = 1

    return tokenized_ent_attr_list

def label_emotion(sentence, annotation_list, tokenizer, emotion_pair):
    temp = {'input_ids': [], 'attention_mask': [], 'token_type_ids' : [], 'label': []}
    for annot in annotation_list:
        ent_attr = annot[0]
        emotion = annot[2]
        tokenized = tokenizer(sentence, ent_attr)
        tokenized.data['label'] = emotion_pair[emotion]
        for key in temp.keys():
            temp[key].append(tokenized.data[key])
    return temp

def preprocess_ent_attr(train_data, dev_data, entity_property_pair, tokenizer):
    train_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'token_type_ids' : []}
    dev_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'token_type_ids' : []}

    print('labeling data...')
    from tqdm import tqdm
    for data in tqdm(train_data):
        temp_data = label_ent_attr(data['sentence_form'], data['annotation'], entity_property_pair, tokenizer)
        train_ent_attr_data = add_dic(train_ent_attr_data, temp_data)
    for data in tqdm(dev_data):
        temp_data = label_ent_attr(data['sentence_form'], data['annotation'], entity_property_pair, tokenizer)
        dev_ent_attr_data = add_dic(dev_ent_attr_data, temp_data)

    return train_ent_attr_data, dev_ent_attr_data

def preprocess_emotion(train_data, dev_data, emotion_pair, tokenizer):
    train_emotion_data = {'input_ids':[], 'attention_mask':[],'label':[], 'token_type_ids' : []}
    dev_emotion_data = {'input_ids':[], 'attention_mask':[],'label':[], 'token_type_ids' : []}

    for data in train_data:
        temp_data = label_emotion(data['sentence_form'], data['annotation'], tokenizer, emotion_pair)
        train_emotion_data = add_dic(train_emotion_data, temp_data)
    for data in dev_data:
        temp_data = label_emotion(data['sentence_form'], data['annotation'], tokenizer, emotion_pair)
        dev_emotion_data = add_dic(dev_emotion_data, temp_data)

    return train_emotion_data, dev_emotion_data

def preprocess_testfile(test_filepath):
    test_dic = jsonl2Diclist(test_filepath)
    return test_dic

# ---------------------- emoji erasing feature ------------------------

def erase_emoji(sentence):
    import re
    preprocessed_sentence = re.sub('[\xa0​ ]', ' ', sentence)
    preprocessed_sentence = re.sub('➕', '+', preprocessed_sentence)
    # preprocessed_sentence = re.sub('[^ A-Za-z0-9가-힣ㄱ-ㅎ!?.,#~*^()+\-:%&\[\]\'\"_/`<>]', '', preprocessed_sentence)
    preprocessed_sentence = re.sub('▲', '', preprocessed_sentence)
    return preprocessed_sentence

def re_annotate(sentence, annotation_list):
    import re
    new_annotation_list = []
    for annotation in annotation_list:
        if annotation[1][0] is None:
            new_annotation_list.append(
                [annotation[0], [annotation[1][0], annotation[1][1], annotation[1][2]], annotation[2]])
        else:
            preprocessed_annotation = erase_emoji(annotation[1][0])
            index = sentence.find(preprocessed_annotation)
            end_index = index + len(annotation[1][0])

            if index == -1 :
                preprocessed_annotation = None
                index = 0
                end_index = 0

            new_annotation_list.append([annotation[0], [preprocessed_annotation, index, end_index], annotation[2]])
    return new_annotation_list

def jsonl2Diclist_erase_emoji(filepath):
    data_dic = jsonl2Diclist(filepath)
    new_data_dic=[]
    for data in data_dic:
        new_sentence = erase_emoji(data['sentence_form'])
        new_annotation = re_annotate(new_sentence, data['annotation'])

        new_data_dic.append({'id':data['id'], 'sentence_form':new_sentence,'annotation': new_annotation})
    return new_data_dic

def preprocess_and_mix_data(train_filepath, dev_filepath, split_ratio=0.9, seed=0):
    dev_dic = jsonl2Diclist_erase_emoji(dev_filepath)
    train_dic = jsonl2Diclist_erase_emoji(train_filepath)

    import random
    if split_ratio != 0:
        dic = train_dic + dev_dic
        random.seed(seed)
        random.shuffle(dic)

        new_data_dic = []
        for data in dic:
            new_sentence = erase_emoji(data['sentence_form'])
            new_annotation = re_annotate(new_sentence, data['annotation'])
            new_data_dic.append({'id': data['id'], 'sentence_form': new_sentence, 'annotation': new_annotation})

        train_dic = dic[:int(len(dic) * split_ratio)]
        dev_dic = dic[int(len(dic) * split_ratio):]

    return train_dic, dev_dic

def label_ent_attr_Addannot(sentence, annotation_list, entity_list, tokenizer):
    tokenized_ent_attr_list = make_ent_attr_data(sentence, entity_list, tokenizer)
    tokenized_ent_attr_list = make_padded_list(tokenized_ent_attr_list, tokenizer)
    tokenized_ent_attr_list['label'] = [0 for _ in range(len(entity_list))]
    for annotation in annotation_list:
        tokenized_ent_attr_list['label'][entity_list.index(annotation[0])] = 1

    for annotation in annotation_list:
        if annotation[1][0] is None : continue
        tokenized_ent_attr_list2 = tokenizer(annotation[1][0], annotation[0])
        tokenized_ent_attr_list2['label'] = 1
        tokenized_ent_attr_list = append_dic(tokenized_ent_attr_list, tokenized_ent_attr_list2.data)
    return tokenized_ent_attr_list
def preprocess_ent_attr_AddAnnot(train_data, dev_data, entity_property_pair, tokenizer):
    train_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'token_type_ids' : []}
    dev_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'token_type_ids' : []}

    print('labeling data...')
    from tqdm import tqdm
    for data in tqdm(train_data):
        temp_data = label_ent_attr_Addannot(data['sentence_form'], data['annotation'], entity_property_pair, tokenizer)
        train_ent_attr_data = add_dic(train_ent_attr_data, temp_data)
    for data in tqdm(dev_data):
        temp_data = label_ent_attr(data['sentence_form'], data['annotation'], entity_property_pair, tokenizer)
        dev_ent_attr_data = add_dic(dev_ent_attr_data, temp_data)

    return train_ent_attr_data, dev_ent_attr_data