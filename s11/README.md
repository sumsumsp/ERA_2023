# Assignment to use GradCam for visualization of trained model on misclassiffied Image 

## Plan
- Use RsNet18
- train resnet18 for 20 epochs on the CIFAR10 dataset
- show loss curves for test and train datasets
- show a gallery of 10 misclassified images
- show gradcam
- Apply these transforms while training: RandomCrop(32, padding=4), CutOut(16x16)
- Test accuracy >85%
- use 

## Depolyed 
### Model Summary
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/766a4432-cdcb-4c9d-aac5-fcd9a5ddc0bc)


![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/a7cd9b4e-ae28-456b-a960-69772d62a8ce)

- Maximum LR
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/f69f74cb-53b6-46d4-a5c6-4d2158565e8f)


Once Cycle LR 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/dbae3643-8997-45fb-acc8-db661c9f692c)

### Training Log 
Epoch 0
loss=1.4921098947525024 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.67it/s]
Train set: Average loss: 0.0033, Accuracy: 18646/50000 (37.29%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -2.8107, Accuracy: 2780/10000 (27.80%)

Epoch 1
loss=1.256839394569397 batch_id=97: 100%|██████████████████████████████████████████████| 98/98 [00:57<00:00,  1.70it/s]
Train set: Average loss: 0.0028, Accuracy: 24140/50000 (48.28%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -3.0574, Accuracy: 3956/10000 (39.56%)

Epoch 2
loss=1.0174195766448975 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.67it/s]
Train set: Average loss: 0.0024, Accuracy: 28025/50000 (56.05%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -2.2917, Accuracy: 4402/10000 (44.02%)

Epoch 3
loss=1.1041933298110962 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:59<00:00,  1.66it/s]
Train set: Average loss: 0.0022, Accuracy: 30217/50000 (60.43%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -1.9631, Accuracy: 3472/10000 (34.72%)

Epoch 4
loss=1.1806488037109375 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.67it/s]
Train set: Average loss: 0.0022, Accuracy: 30518/50000 (61.04%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -2.4702, Accuracy: 3505/10000 (35.05%)

Epoch 5
loss=0.946561336517334 batch_id=97: 100%|██████████████████████████████████████████████| 98/98 [00:58<00:00,  1.69it/s]
Train set: Average loss: 0.0021, Accuracy: 31113/50000 (62.23%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -3.1374, Accuracy: 5116/10000 (51.16%)

Epoch 6
loss=1.0330544710159302 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.67it/s]
Train set: Average loss: 0.0020, Accuracy: 32240/50000 (64.48%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -2.5872, Accuracy: 5064/10000 (50.64%)

Epoch 7
loss=1.1506798267364502 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.67it/s]
Train set: Average loss: 0.0019, Accuracy: 32795/50000 (65.59%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -2.6926, Accuracy: 4754/10000 (47.54%)

Epoch 8
loss=0.7915913462638855 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:57<00:00,  1.69it/s]
Train set: Average loss: 0.0019, Accuracy: 33326/50000 (66.65%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -3.3152, Accuracy: 6256/10000 (62.56%)

Epoch 9
loss=0.7910418510437012 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.69it/s]
Train set: Average loss: 0.0018, Accuracy: 34041/50000 (68.08%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -3.2524, Accuracy: 5873/10000 (58.73%)

Epoch 10
loss=0.8083638548851013 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.68it/s]
Train set: Average loss: 0.0017, Accuracy: 34635/50000 (69.27%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -3.6060, Accuracy: 6081/10000 (60.81%)

Epoch 11
loss=0.7423621416091919 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.69it/s]
Train set: Average loss: 0.0017, Accuracy: 35190/50000 (70.38%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -4.5140, Accuracy: 7264/10000 (72.64%)

Epoch 12
loss=0.8421616554260254 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.68it/s]
Train set: Average loss: 0.0016, Accuracy: 35767/50000 (71.53%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -4.6921, Accuracy: 6886/10000 (68.86%)

Epoch 13
loss=0.7180912494659424 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.68it/s]
Train set: Average loss: 0.0015, Accuracy: 36505/50000 (73.01%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -4.1437, Accuracy: 7134/10000 (71.34%)

Epoch 14
loss=0.701765775680542 batch_id=97: 100%|██████████████████████████████████████████████| 98/98 [00:58<00:00,  1.69it/s]
Train set: Average loss: 0.0015, Accuracy: 36905/50000 (73.81%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -5.6282, Accuracy: 7378/10000 (73.78%)

Epoch 15
loss=0.6289912462234497 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:57<00:00,  1.70it/s]
Train set: Average loss: 0.0014, Accuracy: 37615/50000 (75.23%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -5.1050, Accuracy: 7738/10000 (77.38%)

Epoch 16
loss=0.6157695055007935 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:58<00:00,  1.69it/s]
Train set: Average loss: 0.0013, Accuracy: 38555/50000 (77.11%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -5.8286, Accuracy: 7974/10000 (79.74%)

Epoch 17
loss=0.628288745880127 batch_id=97: 100%|██████████████████████████████████████████████| 98/98 [00:57<00:00,  1.69it/s]
Train set: Average loss: 0.0012, Accuracy: 39221/50000 (78.44%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -6.4597, Accuracy: 8144/10000 (81.44%)

Epoch 18
loss=0.5466952323913574 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:57<00:00,  1.70it/s]
Train set: Average loss: 0.0011, Accuracy: 40455/50000 (80.91%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Test set: Average loss: -6.4851, Accuracy: 8459/10000 (84.59%)

Epoch 19
loss=0.48682326078414917 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:58<00:00,  1.69it/s]
Train set: Average loss: 0.0010, Accuracy: 41604/50000 (83.21%)


Test set: Average loss: -7.4471, Accuracy: 8757/10000 (87.57%)




![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/df83c672-e441-40f8-a78a-f3d9e967d0d1)



### Accuracy of each class 
Accuracy of plane : 94 %
Accuracy of   car : 94 %
Accuracy of  bird : 75 %
Accuracy of   cat : 88 %
Accuracy of  deer : 87 %
Accuracy of   dog : 75 %
Accuracy of  frog : 92 %
Accuracy of horse : 86 %
Accuracy of  ship : 95 %
Accuracy of truck : 91 %

### Misclassiffied images 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/b0f320f3-4ca9-4a8f-95d3-f7d08b474a62)



### Gradcam applied for misclassiffied images 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/20d27b35-432e-4487-96c9-243ee6cc3409)



![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/236b392a-28c9-4519-a95c-e2b743364759)


![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/8b5356df-0b3a-44af-8195-934abde4aed4)


![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/f4990511-b617-4724-8d00-63f41d94a483)


![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/a07ed7d8-dada-4766-b818-aded936d9941)
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/dc0e8b47-5bee-4e94-a2a5-e22be4e115c7)

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/6fd903c9-40d4-412e-9f9b-4e0f03007878)

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/73ba685c-087f-4df9-91f2-072073ed250c)

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/1049fbd8-4fdd-4536-9b90-6ee48f4e83ab)

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/434510d2-4610-4673-ac60-1cd4d4fdd0d0)

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/e0aba065-3383-476f-a151-7df984fe4911)

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/879b5a11-24cc-4bb2-bf0a-013850ef7f78)


### Conclusion 
- Got 88% accuracy in 20 epochs 







