from datasets import load_dataset
from transformers import BertTokenizerFast
import torch
import torch.utils as utils

class ConllDataset:
    def __init__(self, config):
        self.home_path = config['home_path']
        self.model_name = config['model_name']
        self.tokenizer = BertTokenizerFast.from_pretrained(self.home_path + self.model_name)

    def tokenize(self, batch):
        result = {
            'label_ids': [],
            'input_ids': [],
            'token_type_ids': [],
        }
        max_length = 512

        for tokens, label in zip(batch['tokens'], batch['label_ids']):
            tokenids = self.tokenizer(tokens, add_special_tokens=False)

            token_ids = []
            label_ids = [-100]    # [CLS: label]
            for ids, lab in zip(tokenids['input_ids'], label):
                if len(ids) > 1:
                    token_ids.extend(ids)
                    chunk = [-100] * len(ids)
                    chunk[0] = lab
                    label_ids.extend(chunk)
                else:
                    token_ids.extend(ids)
                    label_ids.extend([lab])
            label_ids.append(-100) # [SEP: label]

            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids)
            token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
            result['input_ids'].append(token_ids)
            result['label_ids'].append(label_ids)
            result['token_type_ids'].append(token_type_ids)

        result = self.tokenizer.pad(result, padding='longest', max_length=max_length, return_attention_mask=True)
        for i in range(len(result['input_ids'])):
            diff = len(result['input_ids'][i]) - len(result['label_ids'][i])
            result['label_ids'][i] += [-100] * diff
        return result

    def process(self, input):
        input = input.remove_columns(['id', 'pos_tags', 'chunk_tags'])
        input = input.rename_column('ner_tags', 'label_ids')
        input = input.map(self.tokenize, batched=True, batch_size=len(input))
        input.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label_ids'])
        return input
    
    def get_data(self):
        train_dataset, test_dataset, validation_dateaset = load_dataset(self.home_path + 'data/conll2003', split=['train', 'test', 'validation'])
        train_dataset = self.process(train_dataset)
        test_dataset = self.process(test_dataset)
        validation_dateaset = self.process(validation_dateaset)
        return train_dataset, test_dataset, validation_dateaset

if __name__ == "__main__":
    config = {"home_path": "/Users/tangtony/Documents/stats/Transformers/transformers_ner-main/", "model_name": "bert-base-cased"}
    train_dataset, test_dataset, validation_dataset = ConllDataset(config).get_data()
    
    valid_loader = utils.data.DataLoader(validation_dataset, batch_size=32)
    valid_loader.len
    
