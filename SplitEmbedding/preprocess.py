# -------------------- for split-embedding feature -----------------------
from util.preprocess import jsonl2Diclist, add_dic
import transformers

class SplitEmbedding:
    def __init__(self, entity_property_pair_list, tokenizer : transformers.AutoTokenizer,
                 model : transformers.AutoModel):
        self.entity_property_list = entity_property_pair_list
        self.tokenizer = tokenizer

        # for entity_property_pair in entity_property_pair_list:
        #     splited = entity_property_pair.split('#')
        #     self.entity_list.append(splited[0])
        #     self.property_list.append(splited[1])
        #
        # self.entity_list = list(set(self.entity_list))
        # self.property_list = list(set(self.property_list))
        #
        # tokenized_entity_list = self.tokenizer(self.entity_list, return_tensors='pt', padding=True)
        # hidden = model(tokenized_entity_list['input_ids'])
        # self.entity_embedding_dict={}
        # for n, entity in enumerate(self.entity_list):
        #     self.entity_embedding_dict[entity] = hidden['last_hidden_state'][n, 0, :]

        tokenized_entity_property_list = self.tokenizer(self.property_list, return_tensors='pt', padding=True)
        hidden = model(tokenized_entity_property_list['input_ids'])
        self.entity_property_embedding_list={}
        for n, entity_property_pair in enumerate(self.entity_property_list):
            self.entity_property_embedding_list[entity_property_pair] = hidden['last_hidden_state'][n, 0, :]

    def make_ent_attr_data(self, sentence, entity_property_pair_list):
        # entity_embeddings_list = []
        # property_embeddings_list = []
        entity_property_embeddings_list=[]
        tokenized_sentence_list = self.tokenizer([sentence for _ in range(len(entity_property_pair_list))])

        for entity_property_pair in entity_property_pair_list:
            # entity = entity_property_pair.split('#')[0]
            # property = entity_property_pair.split('#')[1]
            # entity_embeddings_list.append(self.entity_embedding_dict[entity])
            # property_embeddings_list.append(self.property_embedding_dict[property])
            entity_property_embeddings_list.append(self.entity_property_embedding_list[entity_property_pair])

        res= {'input_ids': tokenized_sentence_list['input_ids'], 'attention_mask': tokenized_sentence_list['attention_mask'],
              'entity_property_embeddings':entity_property_embeddings_list}
              # 'entity_embeddings':entity_embeddings_list, 'property_embeddings':property_embeddings_list}
        return res

    def label_and_make_ent_attr_data(self, sentence, annotation_list, entity_property_pair_list):
        tokenized_ent_attr_list = self.make_ent_attr_data(sentence, entity_property_pair_list)
        tokenized_ent_attr_list['label'] = [0 for _ in range(len(entity_property_pair_list))]
        for annotation in annotation_list:
            tokenized_ent_attr_list['label'][entity_property_pair_list.index(annotation[0])] = 1

        return tokenized_ent_attr_list

    def label_emotion(self, sentence, annotation_list, emotion_pair):
        # temp = {'input_ids': [], 'attention_mask': [], 'label': [], 'entity_embeddings': [], 'property_embeddings': []}
        temp = {'input_ids': [], 'attention_mask': [], 'label': [], 'entity_property_embeddings': []}
        for annot in annotation_list:
            ent_attr = annot[0]
            entity_property_embeddings = self.entity_property_embeddings_list[ent_attr]
            # entity = ent_attr.split('#')[0]
            # property = ent_attr.split('#')[1]
            # entity_embeddings = self.entity_embedding_dict[entity]
            # property_embeddings = self.property_embedding_dict[property]

            emotion = annot[2]
            tokenized = self.tokenizer(sentence)
            tokenized.data['label'] = emotion_pair[emotion]

            temp['input_ids'].append(tokenized.data['input_ids'])
            temp['attention_mask'].append(tokenized.data['attention_mask'])
            temp['label'].append(tokenized.data['label'])
            temp['entity_property_embeddings'].append(entity_property_embeddings)
            # temp['entity_embeddings'].append(entity_embeddings)
            # temp['property_embeddings'].append(property_embeddings)
        return temp

    def preprocess_ent_attr_splitembedding(self, train_filepath, dev_filepath, entity_property_pair_list):
        train_dic = jsonl2Diclist(train_filepath)
        dev_dic = jsonl2Diclist(dev_filepath)
        # train_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'entity_embeddings':[], 'property_embeddings':[]}
        # dev_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'entity_embeddings':[], 'property_embeddings':[]}
        train_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'entity_property_embeddings': []}
        dev_ent_attr_data = {'input_ids':[], 'attention_mask':[],'label':[], 'entity_property_embeddings': []}

        print('labeling data...')
        from tqdm import tqdm
        for data in tqdm(train_dic):
            temp_data = self.label_and_make_ent_attr_data(data['sentence_form'], data['annotation'], entity_property_pair_list)
            train_ent_attr_data = add_dic(train_ent_attr_data, temp_data)
        for data in tqdm(dev_dic):
            temp_data = self.label_and_make_ent_attr_data(data['sentence_form'], data['annotation'], entity_property_pair_list)
            dev_ent_attr_data = add_dic(dev_ent_attr_data, temp_data)

        return train_ent_attr_data, dev_ent_attr_data

    def preprocess_emotion_splitembedding(self, train_filepath, dev_filepath, emotion_pair):
        dev_dic = jsonl2Diclist(dev_filepath)
        train_dic = jsonl2Diclist(train_filepath)
        # train_emotion_data = {'input_ids': [], 'attention_mask': [], 'label': [], 'entity_embeddings':[], 'property_embeddings':[]}
        # dev_emotion_data = {'input_ids': [], 'attention_mask': [], 'label': [], 'entity_embeddings':[], 'property_embeddings':[]}
        train_emotion_data = {'input_ids':[], 'attention_mask':[],'label':[], 'entity_property_embeddings': []}
        dev_emotion_data = {'input_ids':[], 'attention_mask':[],'label':[], 'entity_property_embeddings': []}

        for data in train_dic:
            temp_data = self.label_emotion(data['sentence_form'], data['annotation'], emotion_pair)
            train_emotion_data = add_dic(train_emotion_data, temp_data)
        for data in dev_dic:
            temp_data = self.label_emotion(data['sentence_form'], data['annotation'], emotion_pair)
            dev_emotion_data = add_dic(dev_emotion_data, temp_data)

        return train_emotion_data, dev_emotion_data