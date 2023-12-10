import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def dataset():
    # 定义一个图像转换器，将图像转换为张量并归一化
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 使用ImageFolder来读取图片和标签，根据子文件夹的名称自动分配类别
    dataset = torchvision.datasets.ImageFolder(root='img', transform=transform)

    # 使用DataLoader来并行加载数据，每个批次大小为4，打乱顺序
    dataloader = DataLoader(dataset=dataset, batch_size=200, shuffle=True)
    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)
    return dataloader
