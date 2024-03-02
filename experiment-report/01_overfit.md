# Model simple seq2seq
[without data augmentation](https://github.com/Nanoth-T/Senior-Project/tree/main/model/12_SimpleSeq2Seq)

[with augmentation](https://github.com/Nanoth-T/Senior-Project/tree/main/model/14_augmentation)

- original training data 1000 samples
- test set 150 samples
- fixed tempo (80 BPM)
- 4 bars


## without augmentation

**variable**

<img width="122" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/462282d6-14a2-47e7-9995-c37f73925826">

---

**total loss train / test set**

<img width="570" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/e118d6de-ec97-44f8-bc44-395209ea248b">

---

**avg loss train / test set**

<img width="554" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/d7603901-f820-47f4-b51a-e572cf7f9dfc">

---

**avg accuracy train / test set**

<img width="553" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/800234c3-5e34-454c-b6e4-d4ece96af215">

---

**confusion matrix last epoch**

![image](https://github.com/Nanoth-T/Senior-Project/assets/89636847/932470c7-ca18-4ddd-9e29-f20781ec60b7)

---


## with augmentation - 1st try

**variable**

<img width="322" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/eab43171-87ac-42b9-9de9-5f2d86af2c82">


---

**total loss train / test set**

<img width="718" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/7eb1b0c8-7da6-4bd4-94a5-603c661543ab">

---

**avg loss train / test set**

<img width="547" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/6f2f9d0b-e592-41eb-89b2-32b90fb45fbb">

---

**avg accuracy train / test set**

<img width="560" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/3e1d0bbf-37ca-4a79-875b-3281ebdf7a15">

---

**confusion matrix last epoch**

![image](https://github.com/Nanoth-T/Senior-Project/assets/89636847/5fcb969d-4a66-4e06-855d-12b07e65b3c8)


---


## with augmentation 2nd try

**variable**

<img width="303" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/d16f134d-078b-4ea1-9635-24d854922a2c">

---

**total loss train / test set**

<img width="719" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/35d16006-980f-492c-8280-00cebdd82772">

---

**avg loss train / test set**

<img width="548" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/70105684-6ef3-4913-9f3d-406814e2f203">

---

**avg accuracy train / test set**

<img width="563" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/432c19a1-ba83-4ce1-997d-9cef6a70c76a">

---

**confusion matrix last epoch**

![image](https://github.com/Nanoth-T/Senior-Project/assets/89636847/145d2d10-f900-4fbe-ad03-f9cae040998a)


---




## compare all

**total loss train / test set**

<img width="712" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/04e8e3db-6559-4661-ab47-a572c46c86ba">

---

**avg loss train / test set**

<img width="726" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/52d0b85a-e8ac-4231-987a-e27636079f48">

---

**avg accuracy train / test set**

<img width="729" alt="image" src="https://github.com/Nanoth-T/Senior-Project/assets/89636847/d0de1834-19ee-4467-8485-6c1484f1e868">

---
