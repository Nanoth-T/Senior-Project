# Seq2Seq using BahdanauAttention from pytorch tutorial
- [Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.htm)
- with validation set: save model when validate-loss decresed
- without validation set: save last model after training all epochs

## Encoder
- using lstm

## Attention Decoder
- BahdanauAttention
- using lstm


## Training Looop
- 100 epochs
- learning rate 0.005
- hidden size 128
- batch size 10
- n_mels 256
- time_length 200

### Experiment
1. Using training data 01/10 [see details of dataset](https://github.com/Nanoth-T/Senior-Project/blob/0a212eedb03979c1ab93e49b6a01c60145de84d4/Data/!Information.md)
- validation set: data 01/09
- result of last model: overfit
- Acc of training set: 99.94%
- Acc of test set (same as validation set): 17.38%

**result with validation set**
- same as model12 after 3-4 epochs then loss incress


**Accuracy/train**
![Accuracy/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/65e8175d-c064-40f1-98bd-a6a3a4864c03)

**Loss/train**
![Loss/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/d90dc5bb-1f6c-4eed-8fcb-3255ac1117f3)

---

2. Using training data 02/10 [see details of dataset](https://github.com/Nanoth-T/Senior-Project/blob/0a212eedb03979c1ab93e49b6a01c60145de84d4/Data/!Information.md)
- validation set: data 02/09
- result of last model: overfit
- Acc of training set: 100%
- Acc of test set (same as validation set): 17.92%

**result with validation set**
![](https://github.com/Nanoth-T/Senior-Project/assets/89636847/c3ec7051-39e8-4e46-a005-63b26ca888df)

**Accuracy/train**
![Accuracy/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/655f003e-85a8-46a9-9893-316193ea0a05)

**Loss/train**
![Loss/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/f1fabc16-a408-4f3a-a391-8bc71034c781)
