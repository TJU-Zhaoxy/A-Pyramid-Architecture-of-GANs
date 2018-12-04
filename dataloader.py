from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def dataloader0(input_size, batch_size, split='train'):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    dataset = datasets.ImageFolder(root='D:/GAN/data/0/', transform=transform)
    dataloader0 = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader0


def dataloader1(input_size, batch_size, split='train'):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    dataset = datasets.ImageFolder(root='D:/GAN/data/1/', transform=transform)
    dataloader1 = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader1


def dataloader2(input_size, batch_size, split='train'):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    dataset = datasets.ImageFolder(root='D:/GAN/data/2/', transform=transform)
    dataloader2 = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader2


def dataloader3(input_size, batch_size, split='train'):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    dataset = datasets.ImageFolder(root='D:/GAN/data/3/', transform=transform)
    dataloader3 = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader3
