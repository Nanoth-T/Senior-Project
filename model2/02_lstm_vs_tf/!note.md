- add normalization in preprocessing
- add overlap mel spectrogram

- different only in model.py (LSTM vs Transformer)



### with data4note (old data : 256 samples) model run properly
![image](https://github.com/user-attachments/assets/82c69c08-51d5-4c79-b4f0-0219ad687bc2)


### with new data (5 note : 3125 samples) found Nan in Loss
![image](https://github.com/user-attachments/assets/c18df48a-4175-4842-b3b6-6199bae439da)

### acc and loss when training ------
- 1m 21s (Epoch 24/1000), Loss: 192.3879435658455, Accuracy: 0.7586877000457247
- Validate Set, Loss: 112.50015580654144, Accuracy: 0.7455579246624022
Epoch [25/1000]
- 1m 23s (Epoch 25/1000), Loss: 266.30056315660477, Accuracy: 0.685451912818168
- Validate Set, Loss: 117.44522207975388, Accuracy: 0.7269012082444918
Epoch [26/1000]
- 1m 25s (Epoch 26/1000), Loss: nan, Accuracy: 0.6497485139460448
- Validate Set, Loss: nan, Accuracy: 0.0
Epoch [27/1000]
- 1m 30s (Epoch 27/1000), Loss: nan, Accuracy: 0.5
- Validate Set, Loss: nan, Accuracy: 0.0
