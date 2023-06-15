# 7th Assignment: Develop the model modification: <8k Parameters in 3 Steps 

## Target to reach the 99.4% accuracy within 15 epochs with less parameters (<8k)

### Step 0
##### Targets: to reach 99.4% accuracy with 20 epoch 20k 
- Initially took the model from last class (Session 7) and replicated the result


  ![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/4974ef1d-e792-473e-8a5f-ce8463443d50)



##### Result
-99.4% Test Accuracy within 20 Epochs <20K_Parameters 
   - Parameters : 10 k Parameters 
   - Epoch 20
   - Training Acc: 99.13%
   - Test Acc: 99.41%
     
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/debf7e67-fe76-4194-95be-1a475cddaf7e)

#### Analysis
- Not an overfit model (Model_1)
- 15th Epoch reached test accuracy 99.41%
- Used in last layers Global average Pool followed by transition layer


### Step 1
##### Targets: to reach 99.4% accuracy with 15 epoch near 8k parameters  
-- Model modified: (Conv2d_10Kernels_3*3)*twice, Transition (Conv2d_6Kernels_1*1),Maxpool2D, Conv2d_12Kernels, Conv2d_16Kernels, Conv2d_16Kernels, Conv2d_16Kernels, Transition (Conv2d_10Kernels), Adaptive Global Average Pooling
- 8.3k Parameters
  
  ![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/e7497789-e685-4b00-8ab6-e55e7a150d82)

##### Result
-   Epoch 15; TRaining Acc: 98.81; Test Acc: 99.19
 ![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/d44492df-0286-4866-9ed8-0d16c0214f78)
 

#### Analysis
- Not an overfit model (Model_2)
- 12/13th Epochs reached test accuracy 99.33%
- Used in last layers transition layer with Adaptive average Pool
- Training and Test accuracy become plateau  
-  can training make fast 
-  change the learning rate

### Step 2
##### Targets: to reach 99.4% accuracy with 15 epochs near 8k parameters add Learning rate sheduler 
-- Model training modified with learning rate : Step wise increase in the LR  (6th Step with Gamma 0.2) 
- 8.3k Parameters (with same model_2)
  
##### Result
-   Epoch 15; TRaining Acc: 98.90%; Test Acc: 99.45%
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/be849a55-96b6-47e6-8684-e7559b186ded)

#### Analysis
- Not an overfit model (Model_2)
- 13/14th Epochs reached test accuracy 99.45%
- Step increase in training/test at the 6th epoch


### Step 3
##### Targets: to reach 99.4% accuracy with 15 epochs <8k parameters add Learning rate sheduler 
-- Model training modified with learning rate : Step wise increase in the LR  (6th Step with Gamma 0.2) 
- Modified the model parameters- less (7.8k Parameters)

  
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/76e7fb81-2e81-400c-a681-7cd4aee1c6a8)

  
##### Result
-   Epoch 15; TRaining Acc: 98.90%; Test Acc: 99.45%
- 
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/48fa5c7a-7492-4626-954d-a104d8782eba)


#### Analysis
- Not an overfit model (Model_3)
- 7th Epochs reached test accuracy 99.43%
- Step increase in training/test at the 6th epoch
- 13,14,15 th epoch Test accuracy 99.42% 
