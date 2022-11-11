import torch
device = torch.device('cuda:0')

def evaluate_model(log_filename, ent_attr_model, emotion_model , dev_data,
               entity_property_pair, tokenizer, emotion_pair):
    reverse_emotion_pair = dict(map(reversed, emotion_pair.items())) # 편의를 위해 {2:'neutral'} 형태 생성
    ent_attr_model.to(device)   # gpu에 삽입
    emotion_model.to(device)

    from tqdm import tqdm # progress bar 패키지
    import numpy as np
    from util.preprocess import make_ent_attr_data, make_padded_list
    predicted_dev_data = []
    print('evaluating models...')

    with torch.no_grad():
        for data in tqdm(zip(dev_data)):
            # baseline에 따라 (sentence, [x for x in entity-attribute_list] ) 를 tokenize
            ent_attr_data = make_ent_attr_data(data['sentence_form'], entity_property_pair, tokenizer)
            ent_attr_data = make_padded_list(ent_attr_data, tokenizer)
            ent_attr_data['input_ids'] = torch.tensor(ent_attr_data['input_ids']).to(device)
            ent_attr_data['attention_mask'] = torch.tensor(ent_attr_data['attention_mask']).to(device)
            ent_attr_data['token_type_ids'] = torch.tensor(ent_attr_data['token_type_ids']).to(device)

            pred_list = ent_attr_model(**ent_attr_data) # model prediction
            pred_list = np.argmax(pred_list.logits.cpu().detach(), axis=1)  # argmax하여 인덱스 선택
            pred_ent_attr_list = (pred_list == 1).nonzero(as_tuple=True) # entity-attribute list중 해당하는 것의 index 서치

            if len(pred_ent_attr_list[0])!= 0:
                annotation_list=[]
                # predict된 (sentence, entity-attribute)만 선택하여 감성분석
                for pred_ent_attr in pred_ent_attr_list[0].data:
                    sentence_with_pred_ent_attr = ent_attr_data['input_ids'][pred_ent_attr]
                    sentence_with_pred_ent_attr = torch.unsqueeze(sentence_with_pred_ent_attr, dim=0)
                    pred_emotion = emotion_model(sentence_with_pred_ent_attr) # model prediction
                    emotion = np.argmax(pred_emotion.logits[0].cpu().detach()).tolist()
                    annotation = [entity_property_pair[pred_ent_attr], reverse_emotion_pair[emotion]]
                    annotation_list.append(annotation)

                predicted_dev_data.append(
                    {'id': data['id'], 'sentence_form': data['sentence_form'], 'annotation': annotation_list})

            else: # predict된 entity-attribute가 없을 경우 annotation은 비우자
                predicted_dev_data.append({'id':data['id'], 'sentence_form':data['sentence_form'], 'annotation':[]})

    from run_score import evaluation_f1
    import json
    score = evaluation_f1(dev_data, predicted_dev_data)
    with open(log_filename, 'w')as f:
        json.dump(score, f)
    print(score)

    return predicted_dev_data

def evaluate_model_ensemble(log_filename, ent_attr_model_list, emotion_model_list, dev_data,
               entity_property_pair, tokenizer, emotion_pair):


    reverse_emotion_pair = dict(map(reversed, emotion_pair.items()))
    from tqdm import tqdm
    import numpy as np
    predicted_dev_data = []
    from util.preprocess import make_ent_attr_data, make_padded_list

    dummy_pred_ent_attr_list = torch.zeros(len(dev_data),len(entity_property_pair),2)

    for ent_attr_model in ent_attr_model_list:
        ent_attr_model.to(device)
        with torch.no_grad():
            for n, data in enumerate(tqdm(dev_data)):
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
        for n, pred_ent_attr in enumerate(tqdm(pred_ent_attr_list)):
            pred_ent_attr = (pred_ent_attr == 1).nonzero(as_tuple=True)  # entity-attribute list중 해당하는 것의 index 서치
            pred_ent_attr = pred_ent_attr[0].tolist()
            annotation_list = []
            if len(pred_ent_attr)==0:
                predicted_dev_data.append(
                    {'id': dev_data[n]['id'], 'sentence_form': dev_data[n]['sentence_form'], 'annotation': annotation_list})
                continue

            for pred_ent_attr_index in pred_ent_attr:
                ent_attr_data = tokenizer(dev_data[n]['sentence_form'], entity_property_pair[pred_ent_attr_index], return_tensors='pt')
                ent_attr_data = ent_attr_data.to(device)
                dummy_pred_emotion_list = torch.zeros(1, len(emotion_pair))

                for emotion_model in emotion_model_list:
                    emotion_model.to(device)
                    pred_emotion = emotion_model(**ent_attr_data)
                    dummy_pred_emotion_list[:] += (pred_emotion.logits.cpu().detach())/len(emotion_model_list)

                emotion_index = np.argmax(dummy_pred_emotion_list, axis=1)
                annotation = [entity_property_pair[pred_ent_attr_index], reverse_emotion_pair[emotion_index.item()]]
                annotation_list.append(annotation)

            predicted_dev_data.append(
                    {'id': dev_data[n]['id'], 'sentence_form': dev_data[n]['sentence_form'], 'annotation': annotation_list})

    from run_score import evaluation_f1
    import json
    score = evaluation_f1(dev_data, predicted_dev_data)
    with open(log_filename, 'w')as f:
        json.dump(score, f)
    print(score)

    return predicted_dev_data