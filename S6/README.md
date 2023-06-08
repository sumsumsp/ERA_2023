# Part 1
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


# Part2
