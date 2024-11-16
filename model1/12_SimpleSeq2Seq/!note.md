# Simple Seq2Seq model
- with validation set: save model when validate-loss decresed
- without validation set: save last model after training all epochs

## Encoder
- using lstm

## Decoder
- using lstm
- relu

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
- Acc of training set: 100%
- Acc of test set (same as validation set): 24.11%

**result with validation set**
![](https://github.com/Nanoth-T/Senior-Project/assets/89636847/7a90982f-3e43-4be7-8bea-6acca6f33430)


**Accuracy/train**
![Accuracy/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/bc10d5cd-8714-420f-ba11-2fb307b6f0a0)

**Loss/train**
![Loss/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/462cce77-99b4-4d64-b23a-69f782f45ad3)

---

2. Using training data 02/10 [see details of dataset](https://github.com/Nanoth-T/Senior-Project/blob/0a212eedb03979c1ab93e49b6a01c60145de84d4/Data/!Information.md)
- validation set: data 02/09
- result of last model: overfit
- Acc of training set: 100%
- Acc of test set (same as validation set): 16.82%

**result with validation set**
![](https://github.com/Nanoth-T/Senior-Project/assets/89636847/0e4262aa-3dcb-4bb8-813e-013ee4f695d9)

**Accuracy/train**
![Accuracy/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/24242165-e57e-45c4-a1e3-112fedfc741c)

**Loss/train**
![Loss/train](https://github.com/Nanoth-T/Senior-Project/assets/89636847/5d4c6323-944a-4563-90b8-634cf3e2dc47)
