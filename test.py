from transformers import AutoModelForSequenceClassification
import torch, random
device = torch.device('cuda:0')

# evalutate.py와 동일

def make_test_file_ensemble(output_filename,
                   ent_attr_model_list : [],
                    emotion_model_list : [], test_filepath,
                    entity_property_pair, tokenizer, emotion_pair):

    reverse_emotion_pair = dict(map(reversed, emotion_pair.items()))
    from util.preprocess import jsonl2Diclist, jsonl2Diclist_erase_emoji
    test_data = jsonl2Diclist_erase_emoji(test_filepath)

    from tqdm import tqdm
    import numpy as np
    predicted_test_data = []
    from util.preprocess import make_ent_attr_data, make_padded_list

    dummy_pred_ent_attr_list = torch.zeros(len(test_data),len(entity_property_pair),2)

    for ent_attr_model in ent_attr_model_list:
        ent_attr_model.to(device)
        with torch.no_grad():
            for n, data in enumerate(tqdm(test_data)):
                ent_attr_data = make_ent_attr_data(data['sentence_form'], entity_property_pair, tokenizer)
                ent_attr_data = make_padded_list(ent_attr_data, tokenizer)
                ent_attr_data['input_ids'] = torch.tensor(ent_attr_data['input_ids']).to(device)
                ent_attr_data['attention_mask'] = torch.tensor(ent_attr_data['attention_mask']).to(device)
                ent_attr_data['token_type_ids'] = torch.tensor(ent_attr_data['token_type_ids']).to(device)

                pred = ent_attr_model(**ent_attr_data)
                dummy_pred_ent_attr_list[n,:,:] += (pred.logits.cpu().detach())/len(ent_attr_model_list)
            del ent_attr_model

    pred_ent_attr_list = np.argmax(dummy_pred_ent_attr_list, axis=2)

    with torch.no_grad():
        for emotion_model in emotion_model_list:
            emotion_model.to(device)

        for n, pred_ent_attr in enumerate(tqdm(pred_ent_attr_list)):
            pred_ent_attr = (pred_ent_attr == 1).nonzero(as_tuple=True)  # entity-attribute list중 해당하는 것의 index 서치
            pred_ent_attr = pred_ent_attr[0].tolist()
            annotation_list = []
            if len(pred_ent_attr)==0:
                predicted_test_data.append(
                    {'id': test_data[n]['id'], 'sentence_form': test_data[n]['sentence_form'], 'annotation': annotation_list})
                continue

            for pred_ent_attr_index in pred_ent_attr:
                ent_attr_data = tokenizer(test_data[n]['sentence_form'],
                                          entity_property_pair[pred_ent_attr_index], return_tensors='pt')
                ent_attr_data = ent_attr_data.to(device)
                dummy_pred_emotion_list = torch.zeros(1, len(emotion_pair))

                for emotion_model in emotion_model_list:
                    pred_emotion = emotion_model(**ent_attr_data)
                    dummy_pred_emotion_list[:] += (pred_emotion.logits.cpu().detach())/len(emotion_model_list)

                emotion_index = np.argmax(dummy_pred_emotion_list, axis=1)
                annotation = [entity_property_pair[pred_ent_attr_index], reverse_emotion_pair[emotion_index.item()]]
                annotation_list.append(annotation)

            predicted_test_data.append(
                    {'id': test_data[n]['id'], 'sentence_form': test_data[n]['sentence_form'], 'annotation': annotation_list})

    import json
    with open(output_filename,'w')as f:
        json.dump(predicted_test_data, f)
    return predicted_test_data
