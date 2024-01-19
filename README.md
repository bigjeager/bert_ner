# bert_ner
Experimental Reproduction of CoNLL-2003 NER in BERT paper

# USEAGE
1. git clone all the code and data into **WORK\_HOME\_PATH**
2. change **WORK\_HOME\_PATH** to your actual folder
3. download [bert-base-cased](https://huggingface.co/bert-base-cased) ckpt from huggingface to **WORK\_HOME\_PATH**/model_name

# MODELS
| models | type | features |
| --- | --- | --- |
| FT_TOKEN_SOFTMAX | finetune based | first subtoken |
| FEA_LAST_FOUR_CONCAT | feature based | last four hidden concat + BiLSTM |
| FEA_SECOND_TO_LAST | feature based | second2last hidden + BiLSTM |
| FEA_LAST_HIDDEN | feature based | last hidden + BiLSTM |
| FEA_EMBEDDINGS | feature based | first hidden(embedding) + BiLSTM |
| FEA_SUM_LAST_FOUR | feature based | last four hidden sum + BiLSTM |
| FEA_SUM_TWELVE | feature based | all 12 hidden sum + BiLSTM |

# HYPER_PARAMS
- "n_epochs": 10
- "batch_size": 64
- "lr": 3e-4
- "weight_decay": 0.01
- "weight_balance": True

The following results are all based on the hyperparameters above, run only once on 4-V100 with [bert-base-cased](https://huggingface.co/bert-base-cased).

# FineTune based
## FT_TOKEN_SOFTMAX
In the input to BERT, we use a **case-preserving WordPiece** model, and we include the maximal document context provided by the data. Following standard practice, we formulate this as a tagging task but do not use a CRF layer in the output. We use the representation of the **first sub-token** as the input to the token-level classifier over the NER label set.

## Result
| source | f1 | diff |
| --- | --- | --- |
| reproduce | 96.08 | (0.32) |
| paper | 96.4 | |

| type | O | B-PER | I-PER | B-ORG | I-ORG | B-LOC | I-LOC | B-MISC | I-MISC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| precision | 0.9992 | 0.9500 | 0.9900 | 0.8877 | 0.8491 | 0.9697 | 0.8651 | 0.9022 | 0.7850 |
| recall | 0.9914 | 0.9794 | 0.9862 | 0.9545 | 0.9587 | 0.9581 | 0.9728 | 0.9208 | 0.9075 |
| f1 | 0.9953 | 0.9644 | 0.9881 | 0.9199 | 0.9006 | 0.9639 | 0.9158 | 0.9114 | 0.8418 |
| total | 42770 | 1842 | 1307 | 1342 | 751 | 1838 | 257 | 922 | 346 |

# Feature based
we apply the feature-based approach by extracting the activations from one or more layers without fine-tuning any parameters of BERT. These contextual embeddings are used as input to a randomly initialized two-layer 768-dimensional BiLSTM before the classification layer.

## Result
| model | source | f1 | diff |
| --- | --- | --- | --- |
| FEA_LAST_FOUR_CONCAT | reproduce | 95.77 | (0.33) |
| FEA_LAST_FOUR_CONCAT | paper | 96.1 |  |
| FEA_SECOND_TO_LAST | reproduce | 94.93 | (0.67) |
| FEA_SECOND_TO_LAST | paper | 95.6 |  |
| FEA_LAST_HIDDEN | reproduce | 93.88 | (1.02) |
| FEA_LAST_HIDDEN | paper | 94.9 |  |
| FEA_EMBEDDINGS | reproduce | 91.32 | 0.32  |
| FEA_EMBEDDINGS | paper | 91.0 |  |
| FEA_SUM_LAST_FOUR | reproduce | 95.01 | (0.89) |
| FEA_SUM_LAST_FOUR | paper | 95.9 |  |
| FEA_SUM_TWELVE | reproduce | 95.11 | (0.39) |
| FEA_SUM_TWELVE | paper | 95.5 |  |
