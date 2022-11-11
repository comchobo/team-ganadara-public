from transformers import AutoModelForSequenceClassification
import torch
from .preprocess import SplitEmbedding
device = torch.device('cuda:0')

# evalutate.py와 동일
def make_test_file(output_filename,
                   ent_attr_model : AutoModelForSequenceClassification,
                    emotion_model : AutoModelForSequenceClassification, test_filepath,
                    entity_property_pair_list, tokenizer, emotion_pair, splitembedding_class : SplitEmbedding):

    reverse_emotion_pair = dict(map(reversed, emotion_pair.items()))
    ent_attr_model.to(device)
    emotion_model.to(device)

    from util.preprocess import jsonl2Diclist
    test_data = jsonl2Diclist(test_filepath)

    from tqdm import tqdm
    import numpy as np
    predicted_test_data = []

    with torch.no_grad():
        for data in tqdm(test_data):
            # baseline에 따라 (sentence, [x for x in entity-attribute_list] ) 를 tokenize
            ent_attr_data = splitembedding_class.make_ent_attr_data(data['sentence_form'], entity_property_pair_list)
            ent_attr_data['input_ids'] = torch.tensor(ent_attr_data['input_ids']).to(device)
            ent_attr_data['attention_mask'] = torch.tensor(ent_attr_data['attention_mask']).to(device)
            ent_attr_data['entity_property_embeddings'] = torch.stack(ent_attr_data['entity_property_embeddings'], dim=0).to(device)
            # ent_attr_data['entity_embeddings'] = torch.stack(ent_attr_data['entity_embeddings'], dim=0).to(device)
            # ent_attr_data['property_embeddings'] = torch.stack(ent_attr_data['property_embeddings'], dim=0).to(device)

            pred_list = ent_attr_model(**ent_attr_data)  # model prediction
            pred_list = np.argmax(pred_list.logits.cpu().detach(), axis=1)  # argmax하여 인덱스 선택
            pred_ent_attr_list = (pred_list == 1).nonzero(as_tuple=True)  # entity-attribute list중 해당하는 것의 index 서치

            if len(pred_ent_attr_list[0]) != 0:
                annotation_list = []
                # predict된 (sentence, entity-attribute)만 선택하여 감성분석

                for key in ent_attr_data.keys():
                    ent_attr_data[key] = ent_attr_data[key][pred_ent_attr_list[0].data, :]
                pred_emotion = emotion_model(**ent_attr_data)
                emotions = np.argmax(pred_emotion.logits.cpu().detach(), axis=1)
                for n, pred_idex in enumerate(pred_ent_attr_list[0].data):
                    annotation = [entity_property_pair_list[pred_idex], reverse_emotion_pair[int(emotions[n])]]
                    annotation_list.append(annotation)

                predicted_test_data.append(
                    {'id': data['id'], 'sentence_form': data['sentence_form'], 'annotation': annotation_list})

            else:  # predict된 entity-attribute가 없을 경우 annotation은 비우자
                predicted_test_data.append({'id': data['id'], 'sentence_form': data['sentence_form'], 'annotation': []})

    import json
    with open(output_filename,'w')as f:
        json.dump(predicted_test_data, f)
    return predicted_test_data


