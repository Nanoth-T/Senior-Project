# Encoder-Decoder model 
- multiple sequence length with batch
- 3 difference seq_len

## Preprocessing
- use melspectrogram
- signal size for each input (1, 480, 256) *#(n_chanels, time, n_mels)*
- seq_len = (2, 5, 8)
- batch_size = 10 *batch_first=True*
- input shape (10, seq_len, 1, 480//seq_len, 256)
- include SOS and EOS


## Encoder Model
- 1 lstm
- seq_len, flatten signal (batch, seq_len, 480*256//seq_len)
- output, encoder_hidden

## Dcoder Model
- 1 lstm, linear, relu
- input: target one hot shape (batch, 1, 7)
- use previous output as input
- 0% teacher forcing
- use encoder_hidden (context vector)


## Train
- 100 epochs
- learning rate 0.01
- CrossEntropyLoss
- train 10 samples together

## Inference
- set max length to 60
- expected to stop predicting when reaching EOS

## Result
acc: all 100%  --> seq_len = (2, 5, 8)
<img width="960" alt="seq_len=2" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/ae4b0861-e00f-4948-beb2-72c5a6844cfd">
<img width="960" alt="seq_len=5" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/a13fb894-3941-4147-897e-85b565e7684a">
<img width="960" alt="seq_len=8" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/3fb512ba-6864-4799-bbeb-2df5a3f8b8b3">


loss:
![Loss_train (4)](https://github.com/Nanoth-T/Senior-Project/assets/89636847/05cd6b03-3967-4c74-a101-e8da7eeae79f)
blue: seq_len=2, red: seq_len=5, pink: seq_len=8 

## Note
- This is the batch version, and the batch_size cannot be modified --> The code needs to be rewritten to make it more general
- Due to the **split_melspectrogram** function using the logic of -1 to compute multiple seq_len, an error occurs if the time dimension in the mel spectrogram is not divisible evenly, resulting in a remainder of 1
