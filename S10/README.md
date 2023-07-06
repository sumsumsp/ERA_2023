# Assignment find Learning Rate and buid a residual block 

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



![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/7068f358-96c8-432d-bc7c-de7eab58c04f)

Suggested LR: 4.59E-02

Sheduler = OneCycleLR(
    optimizer, 
    max_lr=4.59E-02,
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
loss=1.5238759517669678 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:56<00:00,  1.74it/s]

Train set: Average loss: 0.0038, Accuracy: 18045/50000 (36.09%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 1.3505, Accuracy: 5073/10000 (50.73%)

Epoch 1
loss=1.3157850503921509 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.79it/s]

Train set: Average loss: 0.0027, Accuracy: 25227/50000 (50.45%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 1.1351, Accuracy: 6036/10000 (60.36%)

Epoch 2
loss=1.0671796798706055 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.77it/s]

Train set: Average loss: 0.0023, Accuracy: 29212/50000 (58.42%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.9896, Accuracy: 6608/10000 (66.08%)

Epoch 3
loss=0.8389999866485596 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.79it/s]

Train set: Average loss: 0.0020, Accuracy: 32106/50000 (64.21%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 1.0505, Accuracy: 6669/10000 (66.69%)

Epoch 4
loss=0.8414093255996704 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0018, Accuracy: 34139/50000 (68.28%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.7238, Accuracy: 7539/10000 (75.39%)

Epoch 5
loss=0.7480989098548889 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:56<00:00,  1.75it/s]

Train set: Average loss: 0.0016, Accuracy: 35579/50000 (71.16%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6309, Accuracy: 7841/10000 (78.41%)

Epoch 6
loss=0.7822126746177673 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.77it/s]

Train set: Average loss: 0.0015, Accuracy: 36801/50000 (73.60%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.7210, Accuracy: 7584/10000 (75.84%)

Epoch 7
loss=0.7464777827262878 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.77it/s]

Train set: Average loss: 0.0014, Accuracy: 37829/50000 (75.66%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6698, Accuracy: 7873/10000 (78.73%)

Epoch 8
loss=0.7298762202262878 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0013, Accuracy: 38312/50000 (76.62%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.5753, Accuracy: 8094/10000 (80.94%)

Epoch 9
loss=0.6413329839706421 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:57<00:00,  1.71it/s]

Train set: Average loss: 0.0012, Accuracy: 39020/50000 (78.04%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.5423, Accuracy: 8200/10000 (82.00%)

Epoch 10
loss=0.5772444605827332 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:56<00:00,  1.75it/s]

Train set: Average loss: 0.0012, Accuracy: 39547/50000 (79.09%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.5075, Accuracy: 8339/10000 (83.39%)

Epoch 11
loss=0.6102855205535889 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0011, Accuracy: 40046/50000 (80.09%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4990, Accuracy: 8288/10000 (82.88%)

Epoch 12
loss=0.6206290125846863 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:56<00:00,  1.75it/s]

Train set: Average loss: 0.0011, Accuracy: 40615/50000 (81.23%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4810, Accuracy: 8427/10000 (84.27%)

Epoch 13
loss=0.5219031572341919 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.75it/s]

Train set: Average loss: 0.0010, Accuracy: 41071/50000 (82.14%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4636, Accuracy: 8501/10000 (85.01%)

Epoch 14
loss=0.5092036128044128 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:55<00:00,  1.76it/s]

Train set: Average loss: 0.0010, Accuracy: 41285/50000 (82.57%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4947, Accuracy: 8419/10000 (84.19%)

Epoch 15
loss=0.4842441976070404 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:56<00:00,  1.75it/s]

Train set: Average loss: 0.0009, Accuracy: 41620/50000 (83.24%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.5986, Accuracy: 8211/10000 (82.11%)

Epoch 16
loss=0.4614824652671814 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.79it/s]

Train set: Average loss: 0.0009, Accuracy: 41841/50000 (83.68%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4418, Accuracy: 8599/10000 (85.99%)

Epoch 17
loss=0.43451976776123047 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:58<00:00,  1.66it/s]

Train set: Average loss: 0.0009, Accuracy: 42279/50000 (84.56%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4352, Accuracy: 8697/10000 (86.97%)

Epoch 18
loss=0.4383581280708313 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:54<00:00,  1.79it/s]

Train set: Average loss: 0.0008, Accuracy: 42583/50000 (85.17%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4048, Accuracy: 8735/10000 (87.35%)

Epoch 19
loss=0.44946444034576416 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:54<00:00,  1.80it/s]

Train set: Average loss: 0.0008, Accuracy: 42480/50000 (84.96%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4116, Accuracy: 8691/10000 (86.91%)

Epoch 20
loss=0.46391427516937256 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:53<00:00,  1.82it/s]

Train set: Average loss: 0.0008, Accuracy: 42586/50000 (85.17%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4368, Accuracy: 8615/10000 (86.15%)

Epoch 21
loss=0.5125478506088257 batch_id=97: 100%|█████████████████████████████████████████████| 98/98 [00:56<00:00,  1.74it/s]

Train set: Average loss: 0.0008, Accuracy: 43134/50000 (86.27%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4840, Accuracy: 8615/10000 (86.15%)

Epoch 22
loss=0.37731805443763733 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:58<00:00,  1.67it/s]

Train set: Average loss: 0.0008, Accuracy: 43281/50000 (86.56%)

  0%|                                                                                           | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4060, Accuracy: 8745/10000 (87.45%)

Epoch 23
loss=0.37180882692337036 batch_id=97: 100%|████████████████████████████████████████████| 98/98 [00:54<00:00,  1.79it/s]

Train set: Average loss: 0.0007, Accuracy: 43530/50000 (87.06%)


Test set: Average loss: 0.3930, Accuracy: 8822/10000 (88.22%)


![image](https://github.com/sumsumsp/ERA_2023/assets/77090119/52266165-c8de-4717-b0c2-e5904874f35b)


# Target Reached 88% in 24epochs 

