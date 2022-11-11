from transformers import Trainer
from torch import nn, tensor, device
device = device('cuda:0')

def calculate_class_weights(train_dataset, label_num):
    temp_list = train_dataset.data['label'].to_pylist()
    label_count_list=[]
    for i in range(label_num):
        label_count_list.append(temp_list.count(i))
    class_weight_list = [len(temp_list) / (label_num * label_count) for label_count in label_count_list]
    return class_weight_list

class class_weight_trainer(Trainer):
    def control_class_weights(self, class_weights):
        self.class_weights = tensor(class_weights).to(device)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class LS_trainer(Trainer):
    def control_LS(self, LS):
        self.LS = LS

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.LS)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
