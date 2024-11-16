# 2 lstm model

use [data 02](https://github.com/Nanoth-T/Senior-Project/tree/main/Data/02-very-simple-rhythm-slowver)

## Preprocessing
- use melspectrogram
- signal size for each input (1, 128, 333)
- include SOS and EOS

## Model
- input combined: signal, input (previous output), hidden
- flatten signal (1, 42624)
- input (1, 7)
- hidden (1, 128)
- 2 lstm layer
- use previous output as input for the next output

## Train
- 100 epochs
- learning rate 0.001
- NLLLoss
- train 5 samples together

## Inference
- set max length to 60
- expected to stop predicting when reaching EOS

## Result
✔️ whole, half, quarter, 8th: looks well, stops at EOS

❗ 16th: almost good

<img width="671" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/5df6edbd-9142-4d3b-841e-4aed03a4b0ae">



```
  # forgot to log in to TensorBoard, but the losses are printed out below:

    There are 5 samples in the dataset.
    Epoch [1/100]
    0m 12s (Epoch 1/100), Loss: 37.49983596801758
    Epoch [2/100]
    0m 22s (Epoch 2/100), Loss: 3.8426694869995117
    Epoch [3/100]
    0m 29s (Epoch 3/100), Loss: 1.820089340209961
    Epoch [4/100]
    0m 38s (Epoch 4/100), Loss: 30.411725997924805
    Epoch [5/100]
    0m 47s (Epoch 5/100), Loss: 25.788875579833984
    Epoch [6/100]
    0m 56s (Epoch 6/100), Loss: 20.871660232543945
    Epoch [7/100]
    1m 5s (Epoch 7/100), Loss: 15.91898250579834
    Epoch [8/100]
    1m 13s (Epoch 8/100), Loss: 12.425114631652832
    Epoch [9/100]
    1m 21s (Epoch 9/100), Loss: 9.505017280578613
    Epoch [10/100]
    1m 29s (Epoch 10/100), Loss: 7.572810173034668
    Epoch [11/100]
    1m 37s (Epoch 11/100), Loss: 6.335279941558838
    Epoch [12/100]
    1m 45s (Epoch 12/100), Loss: 5.733296871185303
    Epoch [13/100]
    1m 53s (Epoch 13/100), Loss: 5.302896022796631
    Epoch [14/100]
    2m 1s (Epoch 14/100), Loss: 5.123264312744141
    Epoch [15/100]
    2m 9s (Epoch 15/100), Loss: 4.960993766784668
    Epoch [16/100]
    2m 17s (Epoch 16/100), Loss: 4.656435966491699
    Epoch [17/100]
    2m 25s (Epoch 17/100), Loss: 4.565061569213867
    Epoch [18/100]
    2m 33s (Epoch 18/100), Loss: 4.433730125427246
    Epoch [19/100]
    2m 42s (Epoch 19/100), Loss: 4.521755218505859
    Epoch [20/100]
    2m 51s (Epoch 20/100), Loss: 4.211910247802734
    Epoch [21/100]
    2m 59s (Epoch 21/100), Loss: 4.209683895111084
    Epoch [22/100]
    3m 7s (Epoch 22/100), Loss: 4.100971221923828
    Epoch [23/100]
    3m 15s (Epoch 23/100), Loss: 3.8809852600097656
    Epoch [24/100]
    3m 23s (Epoch 24/100), Loss: 4.227371692657471
    Epoch [25/100]
    3m 30s (Epoch 25/100), Loss: 3.731503486633301
    Epoch [26/100]
    3m 38s (Epoch 26/100), Loss: 3.7399444580078125
    Epoch [27/100]
    3m 47s (Epoch 27/100), Loss: 3.5413198471069336
    Epoch [28/100]
    3m 55s (Epoch 28/100), Loss: 3.663609027862549
    Epoch [29/100]
    4m 3s (Epoch 29/100), Loss: 3.7884788513183594
    Epoch [30/100]
    4m 11s (Epoch 30/100), Loss: 3.4684810638427734
    Epoch [31/100]
    4m 19s (Epoch 31/100), Loss: 3.2871627807617188
    Epoch [32/100]
    4m 27s (Epoch 32/100), Loss: 3.110948324203491
    Epoch [33/100]
    4m 35s (Epoch 33/100), Loss: 3.377613067626953
    Epoch [34/100]
    4m 45s (Epoch 34/100), Loss: 4.680842399597168
    Epoch [35/100]
    4m 54s (Epoch 35/100), Loss: 2.9846396446228027
    Epoch [36/100]
    5m 1s (Epoch 36/100), Loss: 3.0090842247009277
    Epoch [37/100]
    5m 11s (Epoch 37/100), Loss: 3.59637713432312
    Epoch [38/100]
    5m 20s (Epoch 38/100), Loss: 2.8847622871398926
    Epoch [39/100]
    5m 28s (Epoch 39/100), Loss: 2.8719534873962402
    Epoch [40/100]
    5m 35s (Epoch 40/100), Loss: 3.689577102661133
    Epoch [41/100]
    5m 44s (Epoch 41/100), Loss: 3.317214012145996
    Epoch [42/100]
    5m 53s (Epoch 42/100), Loss: 2.8955326080322266
    Epoch [43/100]
    6m 4s (Epoch 43/100), Loss: 3.6469666957855225
    Epoch [44/100]
    6m 13s (Epoch 44/100), Loss: 3.1065077781677246
    Epoch [45/100]
    6m 23s (Epoch 45/100), Loss: 2.9408669471740723
    Epoch [46/100]
    6m 30s (Epoch 46/100), Loss: 2.767836570739746
    Epoch [47/100]
    6m 39s (Epoch 47/100), Loss: 3.301853656768799
    Epoch [48/100]
    6m 47s (Epoch 48/100), Loss: 3.1736152172088623
    Epoch [49/100]
    6m 56s (Epoch 49/100), Loss: 3.1476805210113525
    Epoch [50/100]
    7m 7s (Epoch 50/100), Loss: 3.0133559703826904
    Epoch [51/100]
    7m 17s (Epoch 51/100), Loss: 2.904418706893921
    Epoch [52/100]
    7m 24s (Epoch 52/100), Loss: 2.6408257484436035
    Epoch [53/100]
    7m 32s (Epoch 53/100), Loss: 2.2964601516723633
    Epoch [54/100]
    7m 40s (Epoch 54/100), Loss: 2.3867766857147217
    Epoch [55/100]
    7m 49s (Epoch 55/100), Loss: 2.4173407554626465
    Epoch [56/100]
    7m 57s (Epoch 56/100), Loss: 3.824643135070801
    Epoch [57/100]
    8m 4s (Epoch 57/100), Loss: 2.6934359073638916
    Epoch [58/100]
    8m 12s (Epoch 58/100), Loss: 3.5641374588012695
    Epoch [59/100]
    8m 19s (Epoch 59/100), Loss: 2.8633975982666016
    Epoch [60/100]
    8m 27s (Epoch 60/100), Loss: 2.669691324234009
    Epoch [61/100]
    8m 35s (Epoch 61/100), Loss: 2.626063823699951
    Epoch [62/100]
    8m 43s (Epoch 62/100), Loss: 2.3481507301330566
    Epoch [63/100]
    8m 51s (Epoch 63/100), Loss: 2.9387383460998535
    Epoch [64/100]
    9m 0s (Epoch 64/100), Loss: 3.887423515319824
    Epoch [65/100]
    9m 7s (Epoch 65/100), Loss: 2.9127631187438965
    Epoch [66/100]
    9m 17s (Epoch 66/100), Loss: 2.836364984512329
    Epoch [67/100]
    9m 24s (Epoch 67/100), Loss: 2.917731761932373
    Epoch [68/100]
    9m 31s (Epoch 68/100), Loss: 2.900343418121338
    Epoch [69/100]
    9m 39s (Epoch 69/100), Loss: 2.7477407455444336
    Epoch [70/100]
    9m 48s (Epoch 70/100), Loss: 2.566425323486328
    Epoch [71/100]
    9m 56s (Epoch 71/100), Loss: 2.4235565662384033
    Epoch [72/100]
    10m 4s (Epoch 72/100), Loss: 2.2093918323516846
    Epoch [73/100]
    10m 12s (Epoch 73/100), Loss: 1.9910491704940796
    Epoch [74/100]
    10m 19s (Epoch 74/100), Loss: 1.9476475715637207
    Epoch [75/100]
    10m 27s (Epoch 75/100), Loss: 2.4941563606262207
    Epoch [76/100]
    10m 34s (Epoch 76/100), Loss: 1.868107795715332
    Epoch [77/100]
    10m 42s (Epoch 77/100), Loss: 2.404050588607788
    Epoch [78/100]
    10m 50s (Epoch 78/100), Loss: 2.6426117420196533
    Epoch [79/100]
    10m 57s (Epoch 79/100), Loss: 2.7320876121520996
    Epoch [80/100]
    11m 5s (Epoch 80/100), Loss: 2.776568651199341
    Epoch [81/100]
    11m 18s (Epoch 81/100), Loss: 2.334749460220337
    Epoch [82/100]
    11m 27s (Epoch 82/100), Loss: 2.357090711593628
    Epoch [83/100]
    11m 34s (Epoch 83/100), Loss: 2.185792922973633
    Epoch [84/100]
    11m 43s (Epoch 84/100), Loss: 1.9321759939193726
    Epoch [85/100]
    11m 50s (Epoch 85/100), Loss: 1.7208224534988403
    Epoch [86/100]
    11m 58s (Epoch 86/100), Loss: 1.6230508089065552
    Epoch [87/100]
    12m 5s (Epoch 87/100), Loss: 2.043745994567871
    Epoch [88/100]
    12m 14s (Epoch 88/100), Loss: 4.945774555206299
    Epoch [89/100]
    12m 21s (Epoch 89/100), Loss: 1.847546100616455
    Epoch [90/100]
    12m 31s (Epoch 90/100), Loss: 3.3046107292175293
    Epoch [91/100]
    12m 40s (Epoch 91/100), Loss: 1.8808120489120483
    Epoch [92/100]
    12m 49s (Epoch 92/100), Loss: 2.8993067741394043
    Epoch [93/100]
    12m 56s (Epoch 93/100), Loss: 2.084810256958008
    Epoch [94/100]
    13m 4s (Epoch 94/100), Loss: 2.279568672180176
    Epoch [95/100]
    13m 11s (Epoch 95/100), Loss: 2.0095245838165283
    Epoch [96/100]
    13m 19s (Epoch 96/100), Loss: 1.8988628387451172
    Epoch [97/100]
    13m 27s (Epoch 97/100), Loss: 1.7218713760375977
    Epoch [98/100]
    13m 34s (Epoch 98/100), Loss: 1.5695271492004395
    Epoch [99/100]
    13m 42s (Epoch 99/100), Loss: 1.494949221611023
    Epoch [100/100]
    13m 51s (Epoch 100/100), Loss: 1.724330186843872
    Model trained and stored at model.pth
```
