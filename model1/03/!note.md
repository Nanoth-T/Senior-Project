
model architecture based on [tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

## Preprocessing
- use melspectrogram
- signal size for each input (1, 64, 1220)
- include SOS and EOS

## Model
- input combined: signal, input (previous output), hidden
    - flatten signal (1, 78080)
    - input (1, 7)
    - hidden (1, 128)
- use previous output as input for the next output

## Inference
- set max length to 60
- expected to stop predicting when reaching EOS
<img width="661" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/56f982ce-f877-4dad-bdc2-6328b2b8cee2">

## Result
- loss is decreasing
- **The model cannot stop predicting**

<img width="661" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/a2de6de6-bd08-48fb-bf94-f89b8b997d4c)">
