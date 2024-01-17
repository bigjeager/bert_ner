import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from data import ConllDataset
from models import BertNerSoftmax

path_config = {
    "home_path": "/ossfs/workspace/", 
    "model_path": "models/bert-base-cased",
    "data_path": "data/conll2003"
}

hyper_params = {
    "n_epochs": 5,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "n_steps": 0
}

class BertNer(L.LightningModule):
    def __init__(self, path_config, hyper_params):
        super().__init__()
        self.path_config = path_config
        self.hyper_params = hyper_params
        self.model = BertNerSoftmax.from_pretrained(self.path_config["home_path"] + self.path_config["model_path"], num_labels=9)
        self.validate_pred = []
        self.validate_label = []
        self.automatic_optimization = False
        self.save_hyperparameters()
    
    def save_checkpoint(self):
        self.model.save_pretrained('SAVE_HUGGINGFACE_MODEL')

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        loss, _ , _ = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'], labels=batch['label_ids'])
        self.log("train_loss", loss)

        self.manual_backward(loss)
        opt.step()
        sch.step()
        self.log("lr", sch.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'], labels=batch['label_ids'])
        self.log("validation_loss", loss)
        
        labels = labels.cpu().detach().numpy().flatten()
        preds = np.argmax(logits.cpu().detach().numpy(), axis = -1).flatten()
        self.validate_pred.append(preds)
        self.validate_label.append(labels)
        return loss

    def on_validation_epoch_end(self):
        def flatten(xx):
            return [i for x in xx for i in x ]
        label = flatten(self.validate_label)
        pred = flatten(self.validate_pred)
        print(classification_report(label, pred, digits=4, labels=list(range(1,9))))
        print(confusion_matrix(label, pred, labels=list(range(0, 9))))
        self.validate_label = []
        self.validate_pred = []

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if 'LayerNorm' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": self.hyper_params["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=self.hyper_params["lr"], eps=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.hyper_params["lr"], anneal_strategy='linear', epochs=self.hyper_params["n_epochs"], steps_per_epoch=self.hyper_params["n_steps"])
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()


train_dataset, test_dataset, validation_dataset = ConllDataset(path_config).get_data()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyper_params["batch_size"], shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=hyper_params["batch_size"])
hyper_params["n_steps"] = len(train_loader)
logger = TensorBoardLogger("logs")

is_train = True
if is_train:
    model = BertNer(path_config=path_config, hyper_params=hyper_params)
    trainer = L.Trainer(max_epochs=hyper_params['n_epochs'], logger=logger, log_every_n_steps=10, accelerator='gpu', devices=1)
    trainer.fit(model, train_loader, valid_loader)
    #model.save_checkpoint()
else:
    ret_checkpoint = "colln2003/logs/lightning_logs/version_0/checkpoints/epoch=3-step=1756.ckpt"
    model = BertNer.load_from_checkpoint(path_config['home_path'] + ret_checkpoint, path_config = path_config, hyper_params = hyper_params)
    trainer = L.Trainer(max_epochs=1, logger=logger, log_every_n_steps=10, accelerator='gpu', devices=1)
    trainer.validate(model, valid_loader)
