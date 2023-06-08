# Part 1 Back Propogation Experiment with different learning rates 
### Goal
To do simple neural network training with forward and backward propagation in different learning rates (in excel)

It is a simple neural network with one hidden layer with 2 neurons (h1, h2). The data has come from input neuron layer (i1, i2). It follows same rule as explained above. The features of Neural network are given Figure 1. The detail step is given in Excel sheet attached. 
### Neural Network Architecture 
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/c717bfdc-beac-491d-8c89-e94ebb59e244)

### Feed forward propagation
The data was given (0.05 and 0.1) to neurons which has multiplied with weights (W1 –W4) of neuron (initial weight = 0.15,0.2,0.25,0.3) of first layer and output has been activated using sigmoid function. The output of that activated results will multiply with weights of next layers (output neurons; W5 to W8). The final activation on that output will compared with actual output and loss will be calculated as explained previously. Figure below
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/8ae80bb8-6b1b-4dc4-8625-e86e5dc4ea1f)

### Backward propagation 
The backward propagation means basically updating weights according to the error calculated (actual value – predicted value). The weights will be updated according to error by partial differentiation and chain of rule. Figure below
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/8b252779-cd17-4977-8273-e3f51679fbeb)

### Learning rate experiment
The experiment is performed with different learning late. The learning rate with high values (2.0) reaches the less loss very fast manner. Figure Below
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/829899cd-e961-4236-bda1-a385576bcf6b)


# Part2 Architectural Basics MNIST classification in less than 20K parameters / 20 epochs 
### Modified the code given in MNIST Base Code
### Modification to model
    1) Used nn.sequentual to group convolution and transition blocks
    2) Dropout is added at the end of convolution block
    3) 1x1 convoluttion is used to reduce the channes in transition block
    4) Global average pooling is used 
    6) Fully connected layer is used to convert from 2d to 1d before softmax
    7) In Transition layer using AvgPooling is giving better results than MaxPooling
 
 ### Final Model architecture 
 
 ![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/0b1f4819-0434-4dcf-b7fb-549e615ee2d2)

### Logs 
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Epoch 1
loss=0.12936240434646606 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 62.65it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0525, Accuracy: 9818/10000 (98.18%)

Epoch 2
loss=0.09565366804599762 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 64.46it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0348, Accuracy: 9891/10000 (98.91%)

Epoch 3
loss=0.007405010052025318 batch_id=937: 100%|████████████████████████████████████████| 938/938 [00:14<00:00, 63.51it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0312, Accuracy: 9898/10000 (98.98%)

Epoch 4
loss=0.12059636414051056 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 62.70it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0268, Accuracy: 9908/10000 (99.08%)

Epoch 5
loss=0.024173907935619354 batch_id=937: 100%|████████████████████████████████████████| 938/938 [00:14<00:00, 62.95it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0289, Accuracy: 9898/10000 (98.98%)

Epoch 6
loss=0.00971416849642992 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 64.25it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0233, Accuracy: 9934/10000 (99.34%)

Epoch 7
loss=0.0025206159334629774 batch_id=937: 100%|███████████████████████████████████████| 938/938 [00:14<00:00, 63.82it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0221, Accuracy: 9933/10000 (99.33%)

Epoch 8
loss=0.01180807314813137 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 66.32it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0209, Accuracy: 9936/10000 (99.36%)

Epoch 9
loss=0.3037427067756653 batch_id=937: 100%|██████████████████████████████████████████| 938/938 [00:15<00:00, 61.75it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0231, Accuracy: 9928/10000 (99.28%)

Epoch 10
loss=0.0033745900727808475 batch_id=937: 100%|███████████████████████████████████████| 938/938 [00:14<00:00, 63.74it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0202, Accuracy: 9935/10000 (99.35%)

Epoch 11
loss=0.057400159537792206 batch_id=937: 100%|████████████████████████████████████████| 938/938 [00:14<00:00, 64.53it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0211, Accuracy: 9937/10000 (99.37%)

Epoch 12
loss=0.20171700417995453 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 63.47it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0214, Accuracy: 9934/10000 (99.34%)

Epoch 13
loss=0.15517957508563995 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 63.94it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0215, Accuracy: 9935/10000 (99.35%)

Epoch 14
loss=0.014951188117265701 batch_id=937: 100%|████████████████████████████████████████| 938/938 [00:14<00:00, 65.03it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0210, Accuracy: 9935/10000 (99.35%)

Epoch 15
loss=0.007716956548392773 batch_id=937: 100%|████████████████████████████████████████| 938/938 [00:14<00:00, 63.39it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0206, Accuracy: 9943/10000 (99.43%)

Epoch 16
loss=0.01502101868391037 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 63.46it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0202, Accuracy: 9940/10000 (99.40%)

Epoch 17
loss=0.01797321066260338 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:15<00:00, 61.46it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0215, Accuracy: 9932/10000 (99.32%)

Epoch 18
loss=0.00152197212446481 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 64.28it/s]
  0%|                                                                                          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0197, Accuracy: 9946/10000 (99.46%)

Epoch 19
loss=0.12737925350666046 batch_id=937: 100%|█████████████████████████████████████████| 938/938 [00:14<00:00, 63.20it/s]
Test set: Average loss: 0.0204, Accuracy: 9941/10000 (99.41%)

### Training/Test Accuracy 
Got test accuracy 99.41% (16th Epoch) with gradual decrease in Loss 
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/9a9c5f93-73fd-46d7-93c2-88690da22bd3)

