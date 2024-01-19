import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from data import ConllDataset
from transformers import BertModel
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

path_config = {
    "home_path": "WORK_HOME_PATH", 
    "model_path": "models/bert-base-cased",
    "data_path": "data/conll2003"
}

class_weight = [1.2011924,27.8908795,39.3075746,38.2824143,68.4087883,27.9515778,199.9027237,55.7212581,148.4826590]

class BertNer(L.LightningModule):
    def __init__(self, path_config, hyper_params):
        super().__init__()
        self.path_config = path_config
        self.hyper_params = hyper_params
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.num_labels = self.hyper_params['num_labels']
        self.model_name = self.hyper_params['model_name']
        self.classes = ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-MISC','I-MISC']

        # metrics to compute
        self.multi_class_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_labels, average=None)
        self.multi_class_precision = torchmetrics.classification.MulticlassPrecision(num_classes=self.num_labels, average=None)
        self.multi_class_recall = torchmetrics.classification.MulticlassRecall(num_classes=self.num_labels, average=None)
        self.multi_class_f1_macro = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_labels, average="macro", ignore_index=0)
        self.multi_class_f1_micro = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_labels, average="micro", ignore_index=0)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_labels)
        self.metrics = {
            "f1": self.multi_class_f1,
            "precision": self.multi_class_precision,
            "recall": self.multi_class_recall,
            "macro": self.multi_class_f1_macro,
            "micro": self.multi_class_f1_micro,
            "cm": self.confusion_matrix
        }

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            else:
                pass


        if self.model_name == "FT_TOKEN_SOFTMAX":
            self.bert = BertModel.from_pretrained(self.path_config['home_path'] + self.path_config['model_path'], add_pooling_layer=False)
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
            self.classifier.apply(init_weights)
        elif self.model_name in ("FEA_LAST_FOUR_CONCAT", "FEA_SECOND_TO_LAST", "FEA_LAST_HIDDEN", "FEA_EMBEDDINGS", "FEA_SUM_LAST_FOUR", "FEA_SUM_TWELVE"):
            self.bert = BertModel.from_pretrained(self.path_config['home_path'] + self.path_config['model_path'], add_pooling_layer=False)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
            self.lstm = nn.LSTM(input_size=hidden_size * self.hyper_params['lstm_hidden_scale'], hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=self.bert.config.hidden_dropout_prob, bidirectional=True)
            self.linear = nn.Linear(in_features=hidden_size * 2, out_features=self.num_labels)
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
            self.lstm.apply(init_weights)
            self.linear.apply(init_weights)
        else:
            pass

    def proces_batch(self, batch, batch_idx):
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        token_type_ids=batch['token_type_ids']
        labels=batch['label_ids']
        logits=None
        if self.model_name == "FT_TOKEN_SOFTMAX":
            outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            # (batch_size, seq_len, num_labels) => (batch_size * seq_len, num_labels)
            logits = self.classifier(sequence_output).view(-1, self.num_labels)
        elif self.model_name in ("FEA_LAST_FOUR_CONCAT", "FEA_SECOND_TO_LAST", "FEA_LAST_HIDDEN", "FEA_EMBEDDINGS", "FEA_SUM_LAST_FOUR", "FEA_SUM_TWELVE"):
            outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
            feature = None
            if   self.model_name == "FEA_LAST_FOUR_CONCAT":
                feature = torch.concat(outputs.hidden_states[9:], dim=-1)
            elif self.model_name == "FEA_SECOND_TO_LAST":
                feature = outputs.hidden_states[-2]
            elif self.model_name == "FEA_LAST_HIDDEN":
                feature = outputs.hidden_states[-1]
            elif self.model_name == "FEA_EMBEDDINGS":
                feature = outputs.hidden_states[0]
            elif self.model_name == "FEA_SUM_LAST_FOUR":
                feature = torch.mean(torch.stack(outputs.hidden_states[9:]), dim=0)
            elif self.model_name == "FEA_SUM_TWELVE":
                feature = torch.mean(torch.stack(outputs.hidden_states[1:]), dim=0)
            else:
                pass
            feature = self.dropout(feature)
            hidden_layer = self.lstm(feature)
            logits = self.linear(hidden_layer[0]).view(-1, self.num_labels)
        else:
            pass

        # (batch_size, seq_len) => (batch_size*seq_len,)
        attention_mask = attention_mask.view(-1)
        labels = labels.view(-1)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(self.device) if self.hyper_params['weight_balance'] else None)
        # != -100: other sub_token\[CLS]\[SEP]\[PAD]
        # attention_mask = 1: valid sentense
        mask = (labels != -100) & (attention_mask == 1)
        active_logits = logits[mask, :]
        active_labels = labels[mask]
        loss = loss_fct(active_logits, active_labels)
        return loss, torch.argmax(active_logits, dim=-1).view(-1), active_labels


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        loss, _ , _ = self.proces_batch(batch, batch_idx)
        #self.log("train_loss", loss, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {'train': loss}, self.global_step)

        self.manual_backward(loss)
        opt.step()
        sch.step()
        self.log("lr", sch.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.proces_batch(batch, batch_idx)
        for metric in self.metrics.values():
            metric.update(preds, labels)
        #self.log("validation_loss", loss, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {'validation': loss}, self.global_step) 

    # get log table of f1/precision/recall/total_num
    def log_table_f1_and_cnt(self, f1_total):
        def markdown_line(arr):
            return " | ".join(arr) + " |"
        markdown_text = """
        |  type | %s
        |  --- | %s
        |  precision | %s
        |  recall | %s
        |  f1 | %s
        |  total | %s
        """ % (markdown_line(self.classes), 
               markdown_line([" --- "] * self.num_labels),
               *[markdown_line(f1_total[i, :].numpy().astype('str')) for i in range(f1_total.shape[0])])
        markdown_text = '\n'.join(l.strip() for l in markdown_text.splitlines())
        return markdown_text
    
    # get confusion matrix heatmap
    def log_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        data = pd.DataFrame(
            cm / torch.sum(cm, dim=-1, keepdim=True),
            index=[i for i in self.classes], 
            columns=[i for i in self.classes])
        fig = sn.heatmap(data, annot=True, fmt=".2%").get_figure()
        return fig

    def validation_epoch_end(self, validation_step_outputs):
        f1_map = {}
        for k in self.metrics:
            value = self.metrics[k].compute()
            f1_map[k] = value.cpu()
            self.metrics[k].reset()
        self.logger.experiment.add_scalars('f1', {'micro': f1_map['micro'], 'macro': f1_map['macro']}, self.global_step)

        fig = self.log_confusion_matrix(f1_map['cm'])
        self.logger.experiment.add_figure("confusion matrix", fig, self.current_epoch)

        f1_cnt_table = self.log_table_f1_and_cnt(torch.vstack([f1_map['precision'], f1_map['recall'], f1_map['f1'], torch.sum(f1_map['cm'], dim=-1)]))
        self.logger.experiment.add_text("statistics", f1_cnt_table, self.current_epoch)

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if 'LayerNorm' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hyper_params["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], 
            lr=self.hyper_params["lr"], 
            eps=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
            self.hyper_params["lr"], 
            anneal_strategy='linear', 
            epochs=self.hyper_params["n_epochs"], 
            steps_per_epoch=int(np.ceil(self.hyper_params["n_steps"]/self.trainer.num_devices)))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

if __name__ == "__main__":
    hyper_params = {
        "n_epochs": 10,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "n_steps": 0,
        "num_labels": 9,
        "model_name": "FT_TOKEN_SOFTMAX",
        "weight_balance": True,
        "lstm_hidden_scale": 1
    }

    train_dataset, test_dataset, validation_dataset = ConllDataset(path_config).get_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyper_params["batch_size"], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=hyper_params["batch_size"])
    hyper_params["n_steps"] = len(train_loader)
    logger = TensorBoardLogger("logs", default_hp_metric=False)

    models = {
        "FT_TOKEN_SOFTMAX": 1, 
        "FEA_LAST_FOUR_CONCAT": 4, 
        "FEA_SECOND_TO_LAST": 1, 
        "FEA_LAST_HIDDEN": 1, 
        "FEA_EMBEDDINGS": 1, 
        "FEA_SUM_LAST_FOUR": 1, 
        "FEA_SUM_TWELVE": 1
    }

    is_train = True
    if is_train:
        for model_name in models:
            hyper_params["model_name"] = model_name
            hyper_params["lstm_hidden_scale"] = models[model_name]
            model = BertNer(path_config=path_config, hyper_params=hyper_params)
            trainer = L.Trainer(max_epochs=hyper_params['n_epochs'], logger=logger, log_every_n_steps=1, accelerator='gpu', devices="auto")
            trainer.fit(model, train_loader, valid_loader)
    else:
        ret_checkpoint = "PATH_TO_CKPT_FILE"
        model = BertNer.load_from_checkpoint(path_config['home_path'] + ret_checkpoint, path_config = path_config, hyper_params = hyper_params)
        trainer = L.Trainer(max_epochs=1, logger=logger, log_every_n_steps=1, accelerator='gpu', devices="auto")
        #trainer.validate(model, valid_loader)
        #trainer.test(model, valid_loader)
