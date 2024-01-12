
# use one sample - overfit model

same code as [model/03](https://github.com/Nanoth-T/Senior-Project/tree/404c35888c2a80fe29606c849e1a80051b0b6c6f/model/03)

this is [sample data](https://github.com/Nanoth-T/Senior-Project/tree/404c35888c2a80fe29606c849e1a80051b0b6c6f/Data/02-very-simple-rhythm-slowver)

## Preprocessing
- use melspectrogram
- signal size for each input (1, 64, 666)
- include SOS and EOS

## Model
- input combined: signal, input (previous output), hidden
    - flatten signal (1, 42624)
    - input (1, 7)
    - hidden (1, 128)
- use previous output as input for the next output

## Inference
- set max length to 60
- expected to stop predicting when reaching EOS

## Result
- ✔️ whole, half, quarter: looks well, stops at EOS
- ❗ **8th, 16th**: cannot stop, too many losses

<img width="647" alt="whole, half" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/dcfe9320-a50f-489a-9959-ea6a502a5628">
<img width="674" alt="quarter" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/9fda2517-cc6f-4f90-924d-efa96b88f33d">
<img width="770" alt="8th" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/77526c7b-0598-4ee1-9c27-c5aa9df65bf4">
<img width="767" alt="16th" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/aa169b68-fd70-402d-bcf1-91674c0fcc9c">

## Loss
<img width="767" alt="loss/train" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/1bf29bf6-3f4b-46a2-adb1-4ead4c0695cf">

*(orange-whole, blue-half, red-quarter, skyblue-8th, pink-16th)*
