# Assignment 
- make model C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
- paramters <5k, add one layer, <20 epoch , test accuracy >70%
- Do 3 model with BN, GN, LN

## Exploratory Data Analysis 
--Data Set Features 
-CIFAR10 airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. 
- training/test 50000/10000 images ;size: 32*32*3
- min: [0. 0. 0.]
- max: [1. 1. 1.]
- mean: [0.49139968 0.48215841 0.44653091]
- std: [0.24703223 0.24348513 0.26158784]
- var: [15.56135559 15.11767693 17.44919073]




![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/fd00a6da-4c30-45be-bf43-2baf7822acc8)


## Image preprocessing and augmentation 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/86ea4bd8-e6d8-4c44-8da0-05f95398733b)


  ## Network with batchnormalization
  -- Target:  >70% Accuracy in 20 Epoch ; Add skip connection, <50K parameters  
  
  ![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/56cf4032-1536-4203-9d6f-6db0124d5ba9)

Convolutional Blocks: The model starts with three convolutional blocks, labeled as convblock1a, convblock1b, and convblock1c. Each block consists of a convolutional layer, followed by batch normalization, ReLU activation, and dropout.
convblock1a takes an input image of size 32x32 with 3 channels and applies a 3x3 convolution with 12 output channels.
convblock1b performs another 3x3 convolution on the output of convblock1a, resulting in 24 output channels.
convblock1c applies a final 3x3 convolution on the output of convblock1b, maintaining 24 output channels.
The output of convblock1c is added element-wise to the output of convblock1a, creating skip connections to preserve information.
Transition Block:

The transblock1 consists of a 1x1 convolution that reduces the number of channels from 24 to 12, effectively downsampling the spatial dimensions by half using stride 2. No activation function or normalization is applied.
Pooling:

A max-pooling layer with a 2x2 kernel and stride 2 is applied to reduce the spatial dimensions by half.
Additional Convolutional Blocks and Transition Block:

convblock2a performs a 3x3 convolution on the output of the previous layer, resulting in 24 output channels.
convblock2b applies another 3x3 convolution on the output of convblock2a, maintaining 24 output channels.
convblock2c performs a 3x3 convolution, resulting in 32 output channels.
The transblock2 consists of a 1x1 convolution that reduces the number of channels from 32 to 20 without changing the spatial dimensions.
Pooling:

Another max-pooling layer with a 2x2 kernel and stride 2 is applied, further reducing the spatial dimensions.
Final Convolutional Blocks and Transition Block:

convblock3a performs a 3x3 convolution on the output of the previous layer, resulting in 42 output channels.
convblock3b applies another 3x3 convolution on the output of convblock3a, maintaining 42 output channels.
The transblock3 consists of a 1x1 convolution that reduces the number of channels from 42 to 10, which will be the final number of output channels.
Global Average Pooling:

An adaptive average pooling layer is applied to convert the spatial dimensions of the feature map to a fixed size of 1x1, performing global pooling across the entire feature map.
Output and Activation:

The output of the global average pooling is flattened and passed through a fully connected layer with 10 output units, corresponding to the number of classes.
The log softmax activation function is applied to the output, providing the predicted probabilities for each class.
Overall, this model consists of multiple convolutional blocks with skip connections and pooling layers, aiming to extract hierarchical features from the input image and reduce spatial dimensions. The layer normalization is applied after each convolutional operation to normalize the inputs and stabilize the training process. The dropout layers help in regularizing the model and reduce overfitting.

### Analysis 
Used step LR, SG optimizer 
-- Training Test Log 

0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 0
loss=1.3131154775619507 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.57it/s]

Train set: Train loss: 1303.9275, Train Accuracy: 18813/50000 (37.63%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.5406, Accuracy: 4365/10000 (43.65%)

EPOCH: 1
loss=1.3402862548828125 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.56it/s]

Train set: Train loss: 1063.0445, Train Accuracy: 25292/50000 (50.58%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.2802, Accuracy: 5328/10000 (53.28%)

EPOCH: 2
loss=0.8630286455154419 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:51<00:00, 15.10it/s]

Train set: Train loss: 960.6575, Train Accuracy: 27793/50000 (55.59%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.1690, Accuracy: 5793/10000 (57.93%)

EPOCH: 3
loss=1.1476854085922241 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:51<00:00, 15.16it/s]

Train set: Train loss: 899.4005, Train Accuracy: 29327/50000 (58.65%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.0635, Accuracy: 6223/10000 (62.23%)

EPOCH: 4
loss=1.0038894414901733 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.98it/s]

Train set: Train loss: 861.5449, Train Accuracy: 30135/50000 (60.27%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.0223, Accuracy: 6329/10000 (63.29%)

EPOCH: 5
loss=1.124178171157837 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:51<00:00, 15.18it/s]

Train set: Train loss: 828.0841, Train Accuracy: 31037/50000 (62.07%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.0209, Accuracy: 6343/10000 (63.43%)

EPOCH: 6
loss=0.680099368095398 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:51<00:00, 15.24it/s]

Train set: Train loss: 758.2823, Train Accuracy: 32615/50000 (65.23%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8939, Accuracy: 6845/10000 (68.45%)

EPOCH: 7
loss=1.0392510890960693 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.97it/s]

Train set: Train loss: 744.3860, Train Accuracy: 32884/50000 (65.77%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8634, Accuracy: 6948/10000 (69.48%)

EPOCH: 8
loss=1.1460318565368652 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:51<00:00, 15.16it/s]

Train set: Train loss: 736.2255, Train Accuracy: 33221/50000 (66.44%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8584, Accuracy: 6953/10000 (69.53%)

EPOCH: 9
loss=0.816155731678009 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:51<00:00, 15.17it/s]

Train set: Train loss: 726.3169, Train Accuracy: 33339/50000 (66.68%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8577, Accuracy: 6955/10000 (69.55%)

EPOCH: 10
loss=1.1630102396011353 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.97it/s]

Train set: Train loss: 718.5207, Train Accuracy: 33549/50000 (67.10%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8357, Accuracy: 7019/10000 (70.19%)

EPOCH: 11
loss=0.8996335864067078 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:51<00:00, 15.32it/s]

Train set: Train loss: 716.9933, Train Accuracy: 33609/50000 (67.22%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8440, Accuracy: 7009/10000 (70.09%)

EPOCH: 12
loss=0.7081299424171448 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.96it/s]

Train set: Train loss: 698.6445, Train Accuracy: 34044/50000 (68.09%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8225, Accuracy: 7104/10000 (71.04%)

EPOCH: 13
loss=0.7301122546195984 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.91it/s]

Train set: Train loss: 695.9977, Train Accuracy: 34056/50000 (68.11%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8118, Accuracy: 7129/10000 (71.29%)

EPOCH: 14
loss=1.2489205598831177 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.69it/s]

Train set: Train loss: 693.8265, Train Accuracy: 34137/50000 (68.27%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8198, Accuracy: 7103/10000 (71.03%)

EPOCH: 15
loss=1.1672862768173218 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.86it/s]

Train set: Train loss: 686.8665, Train Accuracy: 34287/50000 (68.57%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8079, Accuracy: 7162/10000 (71.62%)

EPOCH: 16
loss=0.6041986346244812 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.76it/s]

Train set: Train loss: 687.6582, Train Accuracy: 34339/50000 (68.68%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8087, Accuracy: 7147/10000 (71.47%)

EPOCH: 17
loss=1.284946322441101 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:52<00:00, 14.86it/s]

Train set: Train loss: 687.0704, Train Accuracy: 34387/50000 (68.77%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8146, Accuracy: 7129/10000 (71.29%)

EPOCH: 18
loss=0.9888060092926025 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.68it/s]

Train set: Train loss: 684.8223, Train Accuracy: 34368/50000 (68.74%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.7986, Accuracy: 7189/10000 (71.89%)

EPOCH: 19
loss=0.9230596423149109 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:48<00:00, 16.14it/s]

Train set: Train loss: 685.1867, Train Accuracy: 34427/50000 (68.85%)


Test set: Average loss: 0.8095, Accuracy: 7141/10000 (71.41%)


![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/2892d0c2-c5f7-4df9-8b8f-c4ebcfe1b563)

Wrongly diagnosed 
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/2ba32d22-12ff-4cec-9a80-2724214c5593)



Training accuracy of each classes 
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/ed341a06-5653-4b82-ac96-ba2da4837d99)


### Inference 
-  Reached accuracy 71.89% in 20 Epoch ; >70% accuracy in 10th epoch
-  We may have to increase the parameters to increase training and test accuracy
-  Less accuracy for cat classiffication  

## Group Normalization  
-- Added group Normalization in Second block 2b

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/08d3a483-879e-412d-87b2-c3ce887abc53)

### Analysis 
-- Training /Test Log 
0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 0
loss=1.6310980319976807 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.68it/s]

Train set: Train loss: 1308.5078, Train Accuracy: 18583/50000 (37.17%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.5049, Accuracy: 4277/10000 (42.77%)

EPOCH: 1
loss=1.1840691566467285 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.91it/s]

Train set: Train loss: 1057.4111, Train Accuracy: 25416/50000 (50.83%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.2860, Accuracy: 5339/10000 (53.39%)

EPOCH: 2
loss=1.377373456954956 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:52<00:00, 14.86it/s]

Train set: Train loss: 958.6332, Train Accuracy: 27808/50000 (55.62%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.1268, Accuracy: 5947/10000 (59.47%)

EPOCH: 3
loss=0.9042911529541016 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:54<00:00, 14.44it/s]

Train set: Train loss: 893.5253, Train Accuracy: 29506/50000 (59.01%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.0798, Accuracy: 6148/10000 (61.48%)

EPOCH: 4
loss=1.066651463508606 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:53<00:00, 14.69it/s]

Train set: Train loss: 855.3596, Train Accuracy: 30403/50000 (60.81%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9807, Accuracy: 6494/10000 (64.94%)

EPOCH: 5
loss=0.7643267512321472 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.71it/s]

Train set: Train loss: 824.3176, Train Accuracy: 31123/50000 (62.25%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.0187, Accuracy: 6284/10000 (62.84%)

EPOCH: 6
loss=0.930237889289856 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:53<00:00, 14.69it/s]

Train set: Train loss: 759.5230, Train Accuracy: 32670/50000 (65.34%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8770, Accuracy: 6826/10000 (68.26%)

EPOCH: 7
loss=0.9349381327629089 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.64it/s]

Train set: Train loss: 743.2782, Train Accuracy: 33115/50000 (66.23%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8582, Accuracy: 6904/10000 (69.04%)

EPOCH: 8
loss=1.1982375383377075 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:53<00:00, 14.70it/s]

Train set: Train loss: 732.5436, Train Accuracy: 33178/50000 (66.36%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8791, Accuracy: 6873/10000 (68.73%)

EPOCH: 9
loss=1.0701795816421509 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:52<00:00, 14.95it/s]

Train set: Train loss: 730.7740, Train Accuracy: 33275/50000 (66.55%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8371, Accuracy: 6987/10000 (69.87%)

EPOCH: 10
loss=1.088099718093872 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:53<00:00, 14.74it/s]

Train set: Train loss: 719.3762, Train Accuracy: 33533/50000 (67.07%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8443, Accuracy: 6973/10000 (69.73%)

EPOCH: 11
loss=1.058353304862976 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:53<00:00, 14.69it/s]

Train set: Train loss: 716.4191, Train Accuracy: 33619/50000 (67.24%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8328, Accuracy: 6999/10000 (69.99%)

EPOCH: 12
loss=1.9831900596618652 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:54<00:00, 14.39it/s]

Train set: Train loss: 702.4108, Train Accuracy: 33983/50000 (67.97%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8156, Accuracy: 7086/10000 (70.86%)

EPOCH: 13
loss=1.2907583713531494 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:54<00:00, 14.32it/s]

Train set: Train loss: 696.2328, Train Accuracy: 34104/50000 (68.21%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8147, Accuracy: 7081/10000 (70.81%)

EPOCH: 14
loss=0.8888707160949707 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:51<00:00, 15.24it/s]

Train set: Train loss: 698.4567, Train Accuracy: 34039/50000 (68.08%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8224, Accuracy: 7074/10000 (70.74%)

EPOCH: 15
loss=0.9743189215660095 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.90it/s]

Train set: Train loss: 694.2380, Train Accuracy: 34160/50000 (68.32%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8142, Accuracy: 7083/10000 (70.83%)

EPOCH: 16
loss=0.754996657371521 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:48<00:00, 16.15it/s]

Train set: Train loss: 690.4203, Train Accuracy: 34232/50000 (68.46%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8123, Accuracy: 7099/10000 (70.99%)

EPOCH: 17
loss=0.5828442573547363 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:48<00:00, 16.21it/s]

Train set: Train loss: 691.4964, Train Accuracy: 34202/50000 (68.40%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8039, Accuracy: 7114/10000 (71.14%)

EPOCH: 18
loss=1.363530158996582 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:47<00:00, 16.57it/s]

Train set: Train loss: 687.0676, Train Accuracy: 34354/50000 (68.71%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8090, Accuracy: 7099/10000 (70.99%)

EPOCH: 19
loss=1.3218131065368652 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.93it/s]

Train set: Train loss: 690.9045, Train Accuracy: 34274/50000 (68.55%)


Test set: Average loss: 0.8043, Accuracy: 7116/10000 (71.16%)


![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/4d119a32-7b89-41bf-b0e2-6449844d4b1f)


-- Wrongly diagnosed 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/e7932a4b-4bd5-4084-91e8-a89c576ad541)


-- Training accuracy of different classes

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/5e1abd9c-a5b7-446c-9bb6-0857376c0b3a)

### Inference 
- Got similar accuracy Test 70%   test>train 
- Not an overfit model
- Cat has less accuracy


## Adding Layer Normalization 

--Analysis 
-- Added layer normalization in Block 2 adn Block 3 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/77b44738-0ae1-4473-81cb-2175923c673d)



Train log 

 0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 0
loss=1.568773627281189 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:51<00:00, 15.10it/s]

Train set: Train loss: 1308.9268, Train Accuracy: 18502/50000 (37.00%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.4344, Accuracy: 4554/10000 (45.54%)

EPOCH: 1
loss=1.108500599861145 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:50<00:00, 15.43it/s]

Train set: Train loss: 1067.1291, Train Accuracy: 25253/50000 (50.51%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.4397, Accuracy: 4785/10000 (47.85%)

EPOCH: 2
loss=1.5059093236923218 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.94it/s]

Train set: Train loss: 977.3365, Train Accuracy: 27425/50000 (54.85%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.1749, Accuracy: 5810/10000 (58.10%)

EPOCH: 3
loss=1.3720436096191406 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:48<00:00, 15.98it/s]

Train set: Train loss: 931.5166, Train Accuracy: 28547/50000 (57.09%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.1103, Accuracy: 5975/10000 (59.75%)

EPOCH: 4
loss=0.9015898108482361 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:48<00:00, 16.19it/s]

Train set: Train loss: 856.7247, Train Accuracy: 30245/50000 (60.49%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.0328, Accuracy: 6301/10000 (63.01%)

EPOCH: 5
loss=1.5656267404556274 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:48<00:00, 16.02it/s]

Train set: Train loss: 840.1284, Train Accuracy: 30804/50000 (61.61%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 1.0001, Accuracy: 6372/10000 (63.72%)

EPOCH: 6
loss=1.2020971775054932 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:48<00:00, 16.21it/s]

Train set: Train loss: 827.3232, Train Accuracy: 31002/50000 (62.00%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9831, Accuracy: 6454/10000 (64.54%)

EPOCH: 7
loss=1.355547547340393 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:49<00:00, 15.69it/s]

Train set: Train loss: 822.1966, Train Accuracy: 31180/50000 (62.36%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9502, Accuracy: 6596/10000 (65.96%)

EPOCH: 8
loss=1.3969475030899048 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.67it/s]

Train set: Train loss: 796.2322, Train Accuracy: 31818/50000 (63.64%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9421, Accuracy: 6597/10000 (65.97%)

EPOCH: 9
loss=1.12751042842865 batch_id=781: 100%|████████████████████████████████████████████| 782/782 [00:48<00:00, 16.01it/s]

Train set: Train loss: 790.5910, Train Accuracy: 31827/50000 (63.65%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9195, Accuracy: 6683/10000 (66.83%)

EPOCH: 10
loss=0.6551623940467834 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:50<00:00, 15.49it/s]

Train set: Train loss: 786.6489, Train Accuracy: 31952/50000 (63.90%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9184, Accuracy: 6672/10000 (66.72%)

EPOCH: 11
loss=1.7180067300796509 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.79it/s]

Train set: Train loss: 786.2651, Train Accuracy: 31968/50000 (63.94%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9382, Accuracy: 6589/10000 (65.89%)

EPOCH: 12
loss=1.1140803098678589 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:50<00:00, 15.56it/s]

Train set: Train loss: 772.9661, Train Accuracy: 32296/50000 (64.59%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9089, Accuracy: 6699/10000 (66.99%)

EPOCH: 13
loss=1.003919005393982 batch_id=781: 100%|███████████████████████████████████████████| 782/782 [00:50<00:00, 15.62it/s]

Train set: Train loss: 771.0986, Train Accuracy: 32402/50000 (64.80%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9092, Accuracy: 6702/10000 (67.02%)

EPOCH: 14
loss=1.0915510654449463 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.80it/s]

Train set: Train loss: 774.1718, Train Accuracy: 32348/50000 (64.70%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9079, Accuracy: 6694/10000 (66.94%)

EPOCH: 15
loss=0.6675799489021301 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.85it/s]

Train set: Train loss: 769.1315, Train Accuracy: 32317/50000 (64.63%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9201, Accuracy: 6701/10000 (67.01%)

EPOCH: 16
loss=1.0468158721923828 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:50<00:00, 15.59it/s]

Train set: Train loss: 763.9238, Train Accuracy: 32517/50000 (65.03%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9136, Accuracy: 6677/10000 (66.77%)

EPOCH: 17
loss=1.4238080978393555 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:50<00:00, 15.46it/s]

Train set: Train loss: 764.0550, Train Accuracy: 32513/50000 (65.03%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.8981, Accuracy: 6724/10000 (67.24%)

EPOCH: 18
loss=0.8749369978904724 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:49<00:00, 15.65it/s]

Train set: Train loss: 766.7052, Train Accuracy: 32446/50000 (64.89%)

  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.9063, Accuracy: 6703/10000 (67.03%)

EPOCH: 19
loss=1.5543934106826782 batch_id=781: 100%|██████████████████████████████████████████| 782/782 [00:50<00:00, 15.59it/s]

Train set: Train loss: 764.5382, Train Accuracy: 32463/50000 (64.93%)


Test set: Average loss: 0.9055, Accuracy: 6714/10000 (67.14%)




![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/26441f9d-190d-41c0-b292-f0d83c68d514)

​
--Wronlgy classiffied 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/93f23e74-c87c-4761-96c0-39f6c8640fa2)


-- Accuracy of other classess 

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/03838bfa-3b92-4e27-9209-196f5a0d8f8c)



### Inference 
-- less accurate than BN and LN 










  
