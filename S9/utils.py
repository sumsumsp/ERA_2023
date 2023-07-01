import torch
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pylab import *
from torchvision.utils import make_grid, save_image

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
            A.Sequential([A.CropAndPad(px=4, keep_size=False), A.RandomCrop(32,32)]),
            #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            #A.HorizontalFlip(p=0.3),
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1,
                            min_height=16, min_width=16,
                            fill_value=mean, mask_fill_value = None),
            A.Rotate (limit=5, p=0.5),
            A.Normalize(mean, std),
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

    testset = Cifar10SearchDataset(root='./data', train=False,
                                       download=True, transform=test_transform)

    testloader =torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)
    return trainloader, testloader

def getWrongPredictions(model, device, val_loader,classes):
    wrong_idx = []
    wrong_samples = []
    wrong_preds = []
    actual_values = []

    for data,target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        wrong_idx = (pred != target.view_as(pred)).nonzero()[:, 0]
        wrong_samples.append(data[wrong_idx])
        wrong_preds.append(pred[wrong_idx])
        actual_values.append(target.view_as(pred)[wrong_idx])
    return list(zip(torch.cat(wrong_samples),torch.cat(wrong_preds),torch.cat(actual_values)))

def plotWrongPredictions(wrong_predictions,classes):
    fig = plt.figure(figsize=(10,10))
    mean,std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    for i, (sample, wrong_pred, actual_value) in enumerate(wrong_predictions[:20]):
        sample, wrong_pred, actual_value = sample.cpu().numpy(), wrong_pred.cpu(), actual_value.cpu()
        # Undo normalization
        for j in range(sample.shape[0]):
            sample[j] = (sample[j]*std[j])+mean[j]
        sample = np.transpose(sample, (1, 2, 0))
        ax = fig.add_subplot(4, 5, i+1)
        ax.axis('off')
        ax.set_title(f'actual {classes[actual_value.item()]}\npredicted {classes[wrong_pred.item()]}',fontsize=15)
        ax.imshow(sample)
    plt.show()

def imshow(img,c = "" ):    
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title(c)

inv_norm = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))

def plotGradCam(wrong_predictions,cams):
    for i, (sample, wrong_pred, actual_value) in enumerate(wrong_predictions[:20]):
        torch_img = inv_norm(sample)
        normed_torch_img = sample[None]
        images = []
        for gradcam, gradcam_pp in cams:
            mask, _ = gradcam(normed_torch_img)
            heatmap, result = visualize_cam(mask, torch_img)

            mask_pp, _ = gradcam_pp(normed_torch_img)
            heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
            
            images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
            
        grid_image = make_grid(images, nrow=5)
        imshow(grid_image)