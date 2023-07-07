# Assignment find Learning Rate and buid a residual block 
## s10_LR finder_final.ipynb
- Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:
- PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
- Layer1 -
. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
. Add(X, R1)
  
- Layer 2 -
. Conv 3x3 [256k]
. MaxPooling2D, BN, ReLU
- Layer 3 -
. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
. Add(X, R2)
- MaxPooling with Kernel Size 4
- FC Layer 
- SoftMax
- Uses One Cycle Policy such that:
- Total Epochs = 24
- Max at Epoch = 5
- LRMIN = FIND
- LRMAX = FIND
- NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512
-Use ADAM, and CrossEntropyLoss
## Target Accuracy: 90%
## Albumentation code 
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def getCifar10DataLoader(batch_size=512, **kwargs):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    train_transform = A.Compose(
        [
            A.Normalize(mean, std),
            A.Sequential([A.CropAndPad(px=4, keep_size=False), A.RandomCrop(32,32)]),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.HorizontalFlip(p=0.3),
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1,
                            min_height=16, min_width=16,
                            fill_value=mean),
            A.Rotate (limit=5, p=0.5),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )
    trainset = Cifar10SearchDataset(root='./data', train=True,
                                        download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=12)

    testset = Cifar10SearchDataset(root='./data', train=False,
                                       download=True, transform=test_transform)

    testloader =torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=12)
    return trainloader, testloader


## Model Code  
![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/d253646d-6e75-4cbc-8f09-78792960660d)


- model summary
- ![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/6f026cef-a6f4-4ff7-a0f0-d1da002e9731)

## LR Finder code
optimizer = optim.Adam(model.parameters(), lr=0.04, weight_decay= 1e-4)
criterion = nn.CrossEntropyLoss()
lr_finder = LRFinder(model,optimizer,  criterion, device= "cuda" )
lr_finder.range_test (trainloader, end_lr = 10, num_iter =400, step_mode ="exp")
lr_finder.plot()
lr_finder.reset()

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/4e675aba-6eea-4eda-939d-3db6c4bdcf8b)




Suggested LR: 4.03E-02

Sheduler = OneCycleLR(
    optimizer, 
    max_lr=4.03E-02,
    steps_per_epoch= len(trainloader),
    epochs = epochs,
    pct_start =5/epochs,
    div_factor =100, 
    three_phase =False, 
    final_div_factor =100,
    anneal_strategy= 'linear' 
)



## Training log 
 0%|                                                                                           | 0/98 [00:00<?, ?it/s]
Epoch 0
loss=1.5146912336349487 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0041, Accuracy: 16731/50000 (33.46%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 1.7735, Accuracy: 4060/10000 (40.60%)

Epoch 1
loss=1.185870885848999 batch_id=97: 100%|██████████████████████████████████████████████| 98/98 [00:56<00:00,  1.74it/s]

Train set: Average loss: 0.0027, Accuracy: 24806/50000 (49.61%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 1.3196, Accuracy: 5731/10000 (57.31%)

Epoch 2
loss=1.2135556936264038 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.83it/s]

Train set: Average loss: 0.0023, Accuracy: 29167/50000 (58.33%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 1.2121, Accuracy: 5956/10000 (59.56%)

Epoch 3
loss=0.9431464076042175 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.84it/s]

Train set: Average loss: 0.0020, Accuracy: 31876/50000 (63.75%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 1.1639, Accuracy: 6360/10000 (63.60%)

Epoch 4
loss=0.9179015159606934 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.82it/s]

Train set: Average loss: 0.0018, Accuracy: 33452/50000 (66.90%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.8096, Accuracy: 7127/10000 (71.27%)

Epoch 5
loss=1.0155216455459595 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.84it/s]

Train set: Average loss: 0.0017, Accuracy: 34568/50000 (69.14%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.8079, Accuracy: 7204/10000 (72.04%)

Epoch 6
loss=0.7592952847480774 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0016, Accuracy: 35637/50000 (71.27%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.8764, Accuracy: 7105/10000 (71.05%)

Epoch 7
loss=0.730430543422699 batch_id=97: 100%|██████████████████████████████████████████████| 98/98 [00:54<00:00,  1.81it/s]

Train set: Average loss: 0.0016, Accuracy: 36093/50000 (72.19%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.7342, Accuracy: 7587/10000 (75.87%)

Epoch 8
loss=0.7993651032447815 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:56<00:00,  1.74it/s]

Train set: Average loss: 0.0015, Accuracy: 36647/50000 (73.29%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6360, Accuracy: 7881/10000 (78.81%)

Epoch 9
loss=0.7077876925468445 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0015, Accuracy: 36980/50000 (73.96%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6745, Accuracy: 7707/10000 (77.07%)

Epoch 10
loss=0.7080246806144714 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.83it/s]

Train set: Average loss: 0.0014, Accuracy: 37246/50000 (74.49%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6301, Accuracy: 7877/10000 (78.77%)

Epoch 11
loss=0.723296046257019 batch_id=97: 100%|██████████████████████████████████████████████| 98/98 [00:54<00:00,  1.78it/s]

Train set: Average loss: 0.0014, Accuracy: 37663/50000 (75.33%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.7106, Accuracy: 7623/10000 (76.23%)

Epoch 12
loss=0.7428582310676575 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.81it/s]

Train set: Average loss: 0.0014, Accuracy: 37921/50000 (75.84%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6912, Accuracy: 7810/10000 (78.10%)

Epoch 13
loss=0.6044697165489197 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.84it/s]

Train set: Average loss: 0.0013, Accuracy: 38318/50000 (76.64%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6321, Accuracy: 7825/10000 (78.25%)

Epoch 14
loss=0.6938304901123047 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.84it/s]

Train set: Average loss: 0.0013, Accuracy: 38597/50000 (77.19%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6102, Accuracy: 7991/10000 (79.91%)

Epoch 15
loss=0.6378158330917358 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.81it/s]

Train set: Average loss: 0.0012, Accuracy: 39120/50000 (78.24%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6605, Accuracy: 7849/10000 (78.49%)

Epoch 16
loss=0.6483494639396667 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.83it/s]

Train set: Average loss: 0.0012, Accuracy: 39356/50000 (78.71%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.5103, Accuracy: 8279/10000 (82.79%)

Epoch 17
loss=0.5752604603767395 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:53<00:00,  1.83it/s]

Train set: Average loss: 0.0011, Accuracy: 39793/50000 (79.59%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.5619, Accuracy: 8141/10000 (81.41%)

Epoch 18
loss=0.6843724250793457 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.79it/s]

Train set: Average loss: 0.0011, Accuracy: 40302/50000 (80.60%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4632, Accuracy: 8461/10000 (84.61%)

Epoch 19
loss=0.48743900656700134 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:56<00:00,  1.74it/s]

Train set: Average loss: 0.0010, Accuracy: 40857/50000 (81.71%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4784, Accuracy: 8424/10000 (84.24%)

Epoch 20
loss=0.41596609354019165 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0010, Accuracy: 41425/50000 (82.85%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4664, Accuracy: 8461/10000 (84.61%)

Epoch 21
loss=0.3947847783565521 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:57<00:00,  1.71it/s]

Train set: Average loss: 0.0009, Accuracy: 42101/50000 (84.20%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4065, Accuracy: 8628/10000 (86.28%)

Epoch 22
loss=0.3092397153377533 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.77it/s]

Train set: Average loss: 0.0008, Accuracy: 43054/50000 (86.11%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.3377, Accuracy: 8895/10000 (88.95%)

Epoch 23
loss=0.3003857731819153 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.81it/s]

Train set: Average loss: 0.0007, Accuracy: 43884/50000 (87.77%)


Test set: Average loss: 0.3017, Accuracy: 9027/10000 (90.27%)

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/4a0d77bc-5e35-4bdf-a72d-7233d8375c7e)


Learning Rate

![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/607c668d-dfcb-4d24-9881-95ae45b21845)


## Accuracy of classes 
Files already downloaded and verified
Files already downloaded and verified
Accuracy of plane : 100 %
Accuracy of   car : 89 %
Accuracy of  bird : 75 %
Accuracy of   cat : 69 %
Accuracy of  deer : 95 %
Accuracy of   dog : 75 %
Accuracy of  frog : 92 %
Accuracy of horse : 95 %
Accuracy of  ship : 100 %
Accuracy of truck : 83 %
