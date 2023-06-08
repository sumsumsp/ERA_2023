# Part 1
### Goal
To do simple neural network training with forward and backward propagation in different learning rates (in excel)

It is a simple neural network with one hidden layer with 2 neurons (h1, h2). The data has come from input neuron layer (i1, i2). It follows same rule as explained above. The features of Neural network are given Figure 1. The detail step is given in Excel sheet attached. 
## Neural Network Architecture 
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/c717bfdc-beac-491d-8c89-e94ebb59e244)


## Feed forward propagation
The data was given (0.05 and 0.1) to neurons which has multiplied with weights (W1 –W4) of neuron (initial weight = 0.15,0.2,0.25,0.3) of first layer and output has been activated using sigmoid function. The output of that activated results will multiply with weights of next layers (output neurons; W5 to W8). The final activation on that output will compared with actual output and loss will be calculated as explained previously. 

## Backward propagation 
The backward propagation means basically updating weights according to the error calculated (actual value – predicted value). The weights will be updated according to error by partial differentiation and chain of rule. 
