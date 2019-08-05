# Pipeline SPO Extraction Using ERNIE

Pipeline approach for extracting SPO triplets includes two major steps:
- NER: using ERNIE for sequence labeling  -->  `run_sequence_labeling.py`
- Relation Extraction: using ERNIE for classification   -->  `run_classifier.py`


## Dataset

The Baidu SKE dataset has released two portions with SPO annotation: training & dev.

For our experiment, a subset of the original training set is set aside for development purpose, named **dev0**.

Data Statistics:

|      Dataset      |   Purpose   | Sentences | Entities | SPO triplets (relation) |
|:-----------------:|:-----------:|:---------:|:--------:|:-----------------------:|
| train_postag.json |   training  |  151470   |   442k   |          305616         |
|  dev0_postag.json | development |   21639   |   63.3k  |          43650          |
|  dev_postag.json  |     test    |   21639   |   63.4k  |          43749          |


Label Maps:
In `./dataset` keeps a copy of the label maps used for NER and relation extraction.


## Data Preprocessing
Check `preprocess.py`, then execute
```bash
python preprocess.py --data dataset/train_postag.json --output dataset/ner/train.tsv
```

## Finetune and Evaluate Models

A few scripts to show how to use ERNIE:

**Finetune ERNIE for NER**: `./script/run_BaiduSKE_NER.sh`

**Evaluate ERNIE NER**: `./script/eval_BaiduSKE_NER.sh`

- Arguments to modify:
 `--init_checkpoint`, `--do_val`, `--do_test`, `--dev_set`, `--test_set`,  `--num_labels`.

- If you with to save the NER output, add `--do_predict true` in the bash script.


**Finetune ERNIE for RE**: `./script/run_BaiduSKE_relation.sh`

**Evaluate ERNIE RE**: `./script/eval_BaiduSKE_relation.sh`



## Predict
If you already fintuned ERNIE for NER and RE, then you can load the checkpoints and obtain system SPOs.


- Firstly, use `post_process.py` to convert the 3-column NER output (with headers and docids) to a **.tsv** file, which can be used as input for relation prediction.
```bash
python post_process.py --input output/test_conll_output_processed.tsv --output dataset/relation/test_relation.tsv
```


- Then, use `./script/predict_BaiduSKE_relation.sh` to extract SPOs for each document. Make sure that `$TASK_DATA_PATH`, `$CHECKPOINT_PATH` and other arguments are valid directories.
```bash
bash ./script/predict_BaiduSKE_relation.sh
```


- Fianlly, use `eval_spo.py` to compare system SPOs with gold ones, and calculate precision, recall & F1 scores.
```bash
python eval_spo.py --gold dataset/dev_postag.json --system output/test_spo_predicted.json --output output/test_spo_fp.json
```
The `--output` file keeps the wrong SPOs for future error analysis.
