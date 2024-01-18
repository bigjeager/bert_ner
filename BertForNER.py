import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from data import ConllDataset
from transformers import BertModel
import torchmetrics

model_name = "BERT_TOKEN_SOFTMAX"

path_config = {
    "home_path": "/ossfs/workspace/", 
    "model_path": "models/bert-base-cased",
    "data_path": "data/conll2003"
}

hyper_params = {
    "n_epochs": 10,
    "batch_size": 64,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "n_steps": 0,
    "num_labels": 9
}

class BertForNER(L.LightningModule):
    def __init__(self, path_config, hyper_params):
        super().__init__()
        self.path_config = path_config
        self.hyper_params = hyper_params
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.num_labels = self.hyper_params['num_labels']
        self.multi_class_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_labels, average=None, ignore_index=0)
        self.multi_class_f1_macro = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_labels, average="macro", ignore_index=0)
        self.multi_class_f1_micro = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_labels, average="micro", ignore_index=0)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_labels)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if model_name == "BERT_TOKEN_SOFTMAX":
            self.bert = BertModel.from_pretrained(self.path_config['home_path'] + self.path_config['model_path'])
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
            self.classifier.apply(init_weights)

    def proces_batch(self, batch, batch_idx):
        if model_name == "BERT_TOKEN_SOFTMAX":

            input_ids=batch['input_ids']
            attention_mask=batch['attention_mask']
            token_type_ids=batch['token_type_ids']
            labels=batch['label_ids']
            outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            # (batch_size, seq_len, num_labels) => (batch_size * seq_len, num_labels)
            logits = self.classifier(sequence_output).view(-1, self.num_labels)

            # (batch_size, seq_len) => (batch_size*seq_len,)
            attention_mask = attention_mask.view(-1)
            labels = labels.view(-1)

            loss_fct = nn.CrossEntropyLoss()
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
        self.log("train_loss", loss, sync_dist=True)

        self.manual_backward(loss)
        opt.step()
        sch.step()
        self.log("lr", sch.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.proces_batch(batch, batch_idx)
        self.multi_class_f1.update(preds, labels)
        self.multi_class_f1_macro.update(preds, labels)
        self.multi_class_f1_micro.update(preds, labels)
        self.confusion_matrix.update(preds, labels)
        self.log("validation_loss", loss, sync_dist=True)

    def validation_epoch_end(self, validation_step_outputs):
        print( "f1", self.multi_class_f1.compute() )
        print( "macro", self.multi_class_f1_macro.compute() )
        print( "micro", self.multi_class_f1_micro.compute() )
        print( "cm", self.confusion_matrix.compute() )
        self.multi_class_f1.reset()
        self.multi_class_f1_macro.reset()
        self.multi_class_f1_micro.reset()
        self.confusion_matrix.reset()

    def validation_epoch_end_memory_issue(self, validation_step_outputs):
        all_val_out = self.all_gather(validation_step_outputs)
        if self.trainer.is_global_zero:
            labels = []
            preds  = []
            masks = []
            for outputs in all_val_out:
                labels.append(outputs[0])
                preds.append(outputs[1])
                masks.append(outputs[2])
            
            labels = torch.cat(labels).view(-1).cpu().detach().numpy()
            preds = torch.cat(preds).view(-1).cpu().detach().numpy()
            masks = torch.cat(masks).view(-1).cpu().detach().numpy()

            print(classification_report(labels[masks], preds[masks], digits=4, labels=list(range(1,9))))
            print(confusion_matrix(labels[masks], preds[masks], labels=list(range(0, 9))))
        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if 'LayerNorm' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": self.hyper_params["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=self.hyper_params["lr"], eps=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.hyper_params["lr"], anneal_strategy='linear', epochs=self.hyper_params["n_epochs"], steps_per_epoch=self.hyper_params["n_steps"] // 2)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

if __name__ == "__main__":
    train_dataset, test_dataset, validation_dataset = ConllDataset(path_config).get_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyper_params["batch_size"], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=hyper_params["batch_size"])
    hyper_params["n_steps"] = len(train_loader)
    logger = TensorBoardLogger("logs")

    is_train = True
    if is_train:
        model = BertForNER(path_config=path_config, hyper_params=hyper_params)
        trainer = L.Trainer(max_epochs=hyper_params['n_epochs'], logger=logger, log_every_n_steps=10, accelerator='gpu', devices="auto")
        trainer.fit(model, train_loader, valid_loader)

    else:
        ret_checkpoint = "colln2003/logs/lightning_logs/version_3/checkpoints/epoch=0-step=110.ckpt"
        model = BertForNER.load_from_checkpoint(path_config['home_path'] + ret_checkpoint, path_config = path_config, hyper_params = hyper_params)
        trainer = L.Trainer(max_epochs=1, logger=logger, log_every_n_steps=10, accelerator='gpu', devices="auto")
        #trainer.validate(model, valid_loader)
        #trainer.test(model, valid_loader)
