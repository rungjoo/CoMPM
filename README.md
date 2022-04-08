# CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation (NAACL 2022)
![model](./image/model.png)
The overall flow of our model

## Requirements
1. Pytorch 1.8
2. Python 3.6
3. [Transformer 4.4.0](https://github.com/huggingface/transformers)
4. sklearn

## Datasets
Each data is split into train/dev/test in the [dataset folder](https://github.com/rungjoo/CoMPM/tree/master/dataset).
1. [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_publication.htm)
2. [DailyDialog](http://yanran.li/dailydialog.html)
3. [MELD](https://github.com/declare-lab/MELD/)
4. [EmoryNLP](https://github.com/emorynlp/emotion-detection)

## Train
**For CoMPM, CoMPM(s), CoMPM(f)**

Options
- pretrained: type of model (CoM and PM)
- initial: initial weights of the model
- cls: label class
- dataset: one of 4 dataset (dailydialog, EMORY, iemocap, MELD)
- sample: ratio of the number of the train dataset
- freeze: Whether to learn the PM or not

```bash
python3 train.py --initial {pretrained or scratch} --cls {emotion or sentiment} --dataset {dataset} {--freeze}
```

**For a combination of CoM and PM (different model)**

Options
- context_type: type of CoM
- speaker_type: type of PM
```bash
cd CoMPM_diff
python3 train.py {--options}
```

**For CoM or PM**
```bash
cd CoM or PM
python3 train.py {--options}
```

## Testing with pretrained CoMPM
- [Google drive](https://drive.google.com/drive/folders/1VkKygJeI3Qb-kwxMMesFBl7I4uVqGMJF?usp=sharing)
- Unpack model.tar.gz and place it in {dataset}_models/roberta-large/pretrained/no_freeze/{class}/{sampling}/model.bin
    - dataset: dailydialog, EMORY, iemocap, MELD
    - class: "emotion" or "sentiment"
    - sampling: 0.0 ~ 1.0, default: 1.0
    
```bash
python3 test.py
```