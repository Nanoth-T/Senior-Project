# encoder-decoder model


## Preprocessing
- use melspectrogram
- signal size for each input (1, 480, 256) *#(n_chanels, time, n_mels)*
- seq_len = 2
- input shape (2, 1, 240, 256)
- include SOS and EOS

## Encoder Model
- 1 lstm
- seq_len, flatten signal (2, 61440)
- output, encoder_hidden

## Dcoder Model
- 1 lstm, linear, relu
- input: target one hot shape (1, 7)
- use previous output as input
- 0% teacher forcing
- use encoder_hidden (context vector)


## Train
- 100 epochs
- learning rate 0.001
- CrossEntropyLoss
- train 10 samples together

## Inference
- set max length to 60
- expected to stop predicting when reaching EOS

## Result
acc: 100%
![image](https://github.com/Nanoth-T/Senior-Project/assets/89636847/725f43a3-247b-4800-b2db-a324007c6728)



loss:
![Loss_train (3)](https://github.com/Nanoth-T/Senior-Project/assets/89636847/f2d50a2f-86c8-445a-9d32-cb697e5ca805)
