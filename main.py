import os

os.environ['TOKENIZERS_PARALLELISM'] = 'False'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
dev_filepath = './data/nikluge-sa-2022-dev.jsonl'
train_filepath = './data/nikluge-sa-2022-train.jsonl'
test_filepath = './data/nikluge-sa-2022-test.jsonl'

# Baseline code
entity_property_pair = ['제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성',
                        '제품 전체#디자인', '패키지/구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반',
                        '브랜드#일반', '패키지/구성품#다양성', '패키지/구성품#일반', '본품#인지도', '제품 전체#가격',
                        '본품#편의성', '패키지/구성품#편의성', '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질',
                        '제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격']
emotion_pair = {'positive': 0, 'negative': 1, 'neutral': 2}

special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&',
                                  '&bank-account&', '&num&', '&online-account&','★','♥','\'','"']
}

def train_eval_pipeline(pretrained_model_name, output_repo_path, use_class_weight, find_HP, mix_train_dev,
                        HP_dic, ent_attr_lr_list, emo_lr_list, use_LS=0.0, use_Addannot=False):
    from transformers import AutoTokenizer
    tokenizer_name = pretrained_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens(special_tokens_dict)

    from util.preprocess import preprocess_ent_attr, preprocess_emotion, preprocess_and_mix_data, preprocess_ent_attr_AddAnnot

    train_data, dev_data = preprocess_and_mix_data(train_filepath=train_filepath, dev_filepath=dev_filepath,
                                                   split_ratio=mix_train_dev, seed=HP_dic['seed_num'])

    # baseline에 따라 (sentence, [x for x in entity-attribute_list] ) 를 tokenize & True-false annotate 한 데이터 생성
    if use_Addannot==True:
        train_ent_attr_data, dev_ent_attr_data = preprocess_ent_attr_AddAnnot(
            train_data=train_data, dev_data=dev_data,
            entity_property_pair=entity_property_pair, tokenizer=tokenizer)
    else:
        train_ent_attr_data, dev_ent_attr_data = preprocess_ent_attr(
            train_data=train_data, dev_data=dev_data,
            entity_property_pair=entity_property_pair, tokenizer=tokenizer)

    # baseline에 따라 (sentence, entity-attribute) 에 대해 감정을 annotate한 데이터 생성
    train_emotion_data, dev_emotion_data = preprocess_emotion(
        train_data=train_data, dev_data=dev_data, emotion_pair=emotion_pair, tokenizer=tokenizer)

    # Python hash seed. set_seed() 이외의 시드값 고정을 위해 필요
    os.environ['PYTHONHASHSEED'] = str(HP_dic['seed_num'])

    # 학습 코드
    from train import train_pipeline
    HP_set = {'seed_num' : HP_dic['seed_num'],
              'epoch': HP_dic['epoch'],
              'learning_rate': HP_dic['entattr_learning_rate'],
              'batch_size': HP_dic['entattr_batch_size']}

    ent_attr_model = train_pipeline(output_model_name=output_repo_path + '/entattr_model',
                                    LM_model_name=pretrained_model_name,
                                    tokenizer=tokenizer,
                                    train_data=train_ent_attr_data,
                                    dev_data=dev_ent_attr_data,
                                    HP_set=HP_set,
                                    label_num=2,
                                    use_class_weight=use_class_weight,
                                    trial_num=find_HP,
                                    lr1=ent_attr_lr_list[0], lr2=ent_attr_lr_list[1], use_LS=use_LS
                                    )

    HP_set['learning_rate'] = HP_dic['emo_learning_rate']
    HP_set['batch_size'] = HP_dic['emo_batch_size']
    emotion_model = train_pipeline(output_model_name=output_repo_path + '/emo_model',
                                   LM_model_name=pretrained_model_name,
                                   tokenizer=tokenizer,
                                   train_data=train_emotion_data,
                                   dev_data=dev_emotion_data,
                                   HP_set=HP_set,
                                   label_num=3,
                                   use_class_weight=use_class_weight,
                                   trial_num=find_HP,
                                   lr1=emo_lr_list[0], lr2=emo_lr_list[1], use_LS=use_LS
                                   )

    return dev_data, ent_attr_model, emotion_model, tokenizer


if __name__ == '__main__':
    training_HP_set = {'epoch': 10,
                       'entattr_learning_rate': 1.2163453246352856e-05,
                       'entattr_batch_size': 24,
                       'emo_learning_rate': 1.7064278380264195e-05,
                       'emo_batch_size': 24}
    lmkor = 'kykim/electra-kor-base'

    # mix_train_dev는 0~1 사이의 비율을 넣는다. 0.8일 시 train과 dev를 합치고 80%만큼 train 데이터로 사용. 0일 시 mix하지 않음
    # find_HP는 optuna HP 찾기 기능에서 iteration 횟수. 0일 시 HP를 찾지 않는다.
    seed_list = [0, 123, 12345, 960126, 980107, 11, 7777, 111, 5678, 777]
    for seed in seed_list:
        training_HP_set['seed_num'] = seed
        output_repo_path = './models/baseline-lmkor-seed' + str(seed)
        dev_data, ent_attr_model, emotion_model, tokenizer = \
            train_eval_pipeline(pretrained_model_name=lmkor,
                                output_repo_path=output_repo_path, use_class_weight=True, find_HP=0, mix_train_dev=0.9,
                                HP_dic=training_HP_set, ent_attr_lr_list=[4e-6, 1.5e-5]
                                , emo_lr_list=[5e-6, 3e-5], use_LS=0, use_Addannot=False)

    # baseline 코드에 따라 train data를 추론하고, dev data와 비교하여 score 출력
    # from evaluate import evaluate_model, evaluate_model_ensemble
    #
    # evaluate_model_ensemble(log_filename=output_repo_path + '/eval_score.json',
    #                ent_attr_model_list=[ent_attr_model],
    #                emotion_model_list=[emotion_model], dev_data=dev_data,
    #                entity_property_pair=entity_property_pair, emotion_pair=emotion_pair,
    #                tokenizer=tokenizer
    #                )



