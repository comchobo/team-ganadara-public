from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == '__main__':
    lmkor = 'kykim/electra-kor-base'
    tokenizer = AutoTokenizer.from_pretrained(lmkor)
    ent_attr_model_list = []
    seed_list = [123,12345,11,7777,5678,777,111]
    for seed in seed_list:
        ent_attr_model = AutoModelForSequenceClassification.from_pretrained(
            'sorryhyun/koreancomp2022-entattr-model-seed-'+str(seed))
        ent_attr_model.resize_token_embeddings(len(tokenizer))
        ent_attr_model_list.append(ent_attr_model)

    emotion_model_list = []
    seed_list = [0,123,960126,980107,7777,777,111]
    for seed in seed_list:
        emotion_model = AutoModelForSequenceClassification.from_pretrained(
            'sorryhyun/koreancomp2022-emo-model-seed-'+str(seed))
        emotion_model.resize_token_embeddings(len(tokenizer))
        emotion_model_list.append(emotion_model)

    test_filepath = './data/nikluge-sa-2022-test.jsonl'
    entity_property_pair = ['제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성',
                            '제품 전체#디자인', '패키지/구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반',
                            '브랜드#일반', '패키지/구성품#다양성', '패키지/구성품#일반', '본품#인지도', '제품 전체#가격',
                            '본품#편의성', '패키지/구성품#편의성', '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질',
                            '제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격']
    emotion_pair = {'positive': 0, 'negative': 1, 'neutral': 2}

    # baseline 코드에 따라 test data를 추론하고 json으로 출력. 모델 리스트를 삽입하면 자동으로 앙상블이 된다.
    from test import make_test_file_ensemble
    make_test_file_ensemble(output_filename='./submission.json',
                            ent_attr_model_list=ent_attr_model_list,
                            emotion_model_list=emotion_model_list, test_filepath=test_filepath,
                            entity_property_pair=entity_property_pair,
                            tokenizer=tokenizer, emotion_pair=emotion_pair)