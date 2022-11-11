import json
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, ElectraForSequenceClassification
import numpy as np
from datasets import Dataset, load_metric
from transformers import set_seed

def train_pipeline(output_model_name, LM_model_name, tokenizer, train_data, dev_data, HP_set, label_num, trial_num,
                   lr1, lr2,
                   use_class_weight=False, use_LS=0):
    set_seed(HP_set['seed_num'])

    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(LM_model_name, num_labels=label_num)
        model.resize_token_embeddings(len(tokenizer))
        return model
    model = AutoModelForSequenceClassification.from_pretrained(LM_model_name, num_labels=label_num)
    model.resize_token_embeddings(len(tokenizer))

    from class_weight import calculate_class_weights
    class_weights = calculate_class_weights(train_dataset, label_num)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if label_num == 2 : average = 'binary'
    else: average = 'macro'

    def compute_metrics(eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels, average=average)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    from transformers import TrainingArguments, Trainer
    bestrun = None
    repo_name = output_model_name

    if trial_num > 0:
        training_args = TrainingArguments(
            output_dir=repo_name,
            num_train_epochs=HP_set['epoch'],
            weight_decay=0.01,
            push_to_hub=False,
            dataloader_num_workers=12,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            metric_for_best_model='eval_f1'
        )
        if use_class_weight == True:
            from class_weight import class_weight_trainer
            trainer = class_weight_trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            trainer.control_class_weights(class_weights)

        else:
            from class_weight import LS_trainer
            trainer = LS_trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            trainer.control_LS(use_LS)

        def hp_space_ray(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", lr1, lr2, log=True),
                "per_device_train_batch_size": trial.suggest_categorical('per_device_train_batch_size', [16, 24, 32])
            }

        metric_accumul=[]
        def target_object(metrics):
            if metrics['epoch'] == 1 :
                metric_accumul.clear()
            metric_accumul.append(metrics['eval_f1'])
            return max(metric_accumul)

        bestrun = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=hp_space_ray,
            compute_objective=target_object,
            n_trials=trial_num
        )
        with open('./'+repo_name+'/found_HP_'+bestrun.run_id+'.json','w')as f:
            json.dump(bestrun.hyperparameters, f)

        import os
        import shutil
        folderlist = os.listdir('./'+repo_name)
        for folder in folderlist:
            if folder.find('run-') != -1:
                shutil.rmtree('./' + repo_name + '/' + folder)

    # ------------------ after HP search -----------------------

    training_args = TrainingArguments(
        output_dir=repo_name,
        num_train_epochs=HP_set['epoch'],
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_num_workers=12,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        learning_rate=HP_set['learning_rate'] if trial_num==0 else bestrun.hyperparameters['learning_rate'],
        per_device_train_batch_size=HP_set['batch_size'] if trial_num==0 else bestrun.hyperparameters[
            'per_device_train_batch_size'],
        per_device_eval_batch_size=60,
        metric_for_best_model='eval_f1'
    )

    if use_class_weight == True:
        from class_weight import class_weight_trainer
        trainer = class_weight_trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.control_class_weights(class_weights)

    else:
        from class_weight import LS_trainer
        trainer = LS_trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.control_LS(use_LS)

    print('training models...')
    trainer.train()
    return trainer.model

