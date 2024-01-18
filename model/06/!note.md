# 2 lstm model

use [data 03](https://github.com/Nanoth-T/Senior-Project/tree/main/Data/03-rhythm20sample)

## Preprocessing
- use melspectrogram
- signal size for each input (1, 256, 480)
- include SOS and EOS

## Model
- input combined: signal, input (previous output), hidden
- flatten signal (1, 122880)
- input (1, 7)
- hidden (1, 128)
- 2 lstm layer
- use previous output as input for the next output

## Train
- 100 epochs
- learning rate 0.001
- NLLLoss
- train 20 samples together

## Inference
- set max length to 60
- expected to stop predicting when reaching EOS

## Result
wrong one sample

acc: 96.92%

<img width="667" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/1b30a144-a738-48d6-9f9f-1b813b242322">

loss:

![Loss_train (2)](https://github.com/Nanoth-T/Senior-Project/assets/89636847/5629bf64-66c3-468c-b890-9e2df48fb364)
