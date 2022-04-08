# CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation (NAACL 2022)
![model](./image/model.png)
The overall flow of our model

## Requirements
1. Pytorch 1.8
2. Python 3.6
3. [Transformer 4.4.0](https://github.com/huggingface/transformers)
4. sklearn

## Datasets
Each data is split into train/dev/test in the dataset folder.
1. [IEMOCAP](https://github.com/lijuncen/Sentiment-and-Style-Transfer)
2. [DailyDialog](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data)
3. [MELD](https://github.com/luofuli/DualRL/tree/master/references)
4. [EmoryNLP]()

## Train
For CoMPM, CoMPM(s), CoMPM(f)
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

For a combination of CoM and PM (different model)
Options
- context_type: type of CoM
- speaker_type: type of PM
```bash
cd CoMPM_diff
python3 train.py {--options}
```

For CoMPM
```bash
cd CoMPM
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