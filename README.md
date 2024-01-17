# bert_ner
Experimental Reproduction of CoNLL-2003 NER in BERT paper

# bert_ner_finetune.py
In the input to BERT, we use a **case-preserving WordPiece** model, and we include the maximal document context provided by the data. Following standard practice, we formulate this as a tagging task but do not use a CRF layer in the output. We use the representation of the **first sub-token** as the input to the token-level classifier over the NER label set.

## Best Result
| type | f1 | val |
| --- | --- | --- |
| self | dev-micro-f1 | 96.02 |
| paper | dev-f1 | 96.4 |

```
hyper_params:
  batch_size: 64
  lr: 0.0001
  n_epochs: 8
  weight_decay: 0.01
```
|  | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| 1 | 0.9788 | 0.9777 | 0.9783 | 1842 |
| 2 | 0.9885 | 0.9862 | 0.9874 | 1307 |
| 3 | 0.9387 | 0.9485 | 0.9436 | 1341 |
| 4 | 0.9430 | 0.9467 | 0.9449 | 751 |
| 5 | 0.9748 | 0.9690 | 0.9719 | 1837 |
| 6 | 0.9531 | 0.9494 | 0.9513 | 257 |
| 7 | 0.9216 | 0.9306 | 0.9261 | 922 |
| 8 | 0.8911 | 0.8988 | 0.8950 | 346 |
|   micro avg | 0.9595 | 0.9608 | 0.9602 | 8603 |
|   macro avg | 0.9487 | 0.9509 | 0.9498 | 8603 |
|weighted avg | 0.9596 | 0.9608 | 0.9602 | 8603 |
