import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_root = './data/90'
batch_size = 16
epoch_size = 10
learning_rate = 0.0001
weight_decay = 1e-4

# 可选: 'none', 'channel', 'spatial', 'cbam'
attention_type = 'spatial'

model_path = f'ResNet18_Rafd90_{attention_type}.pth'

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class_names = datasets.ImageFolder(root=dataset_root).classes
num_classes = len(class_names)


def denormalize(image):
    image = image.cpu() * imagenet_std + imagenet_mean
    return image.clamp(0, 1)


def split_indices_by_person(dataset, train_ratio=0.8, seed=42):
    person_to_indices = {}
    for index, (path, label) in enumerate(dataset.samples):
        filename = os.path.basename(path)
        person_id = '_'.join(filename.split('_')[:2])
        person_to_indices.setdefault(person_id, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    persons = list(person_to_indices.keys())
    random_index = torch.randperm(len(persons), generator=generator).tolist()
    train_person_size = int(train_ratio * len(persons))

    train_persons = {persons[i] for i in random_index[:train_person_size]}
    train_indices = []
    test_indices = []
    for person_id, indices in person_to_indices.items():
        if person_id in train_persons:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return train_indices, test_indices


def load_dataset():
    train_dataset = datasets.ImageFolder(root=dataset_root, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=dataset_root, transform=test_transform)
    train_indices, test_indices = split_indices_by_person(train_dataset)
    return Subset(train_dataset, train_indices), Subset(test_dataset, test_indices)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_attention = None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        self.last_attention = attention.detach()
        return x * attention


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.last_attention = None

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        self.last_attention = attention.detach()
        return x * attention


class AttentionBlock(nn.Module):
    def __init__(self, channels, attention='cbam'):
        super().__init__()
        self.attention = attention
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        if self.attention == 'channel':
            x = self.channel_attention(x)
        elif self.attention == 'spatial':
            x = self.spatial_attention(x)
        elif self.attention == 'cbam':
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
        return x


class ResNetBasicBlockAttention(nn.Module):
    expansion = 1

    def __init__(self, old_block, attention='cbam'):
        super().__init__()
        self.conv1 = old_block.conv1
        self.bn1 = old_block.bn1
        self.relu = old_block.relu
        self.conv2 = old_block.conv2
        self.bn2 = old_block.bn2
        self.downsample = old_block.downsample
        self.stride = old_block.stride
        self.attention = AttentionBlock(self.bn2.num_features, attention)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18AttentionNet(nn.Module):
    def __init__(self, attention='none'):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.attention_blocks = nn.ModuleList()

        if attention != 'none':
            self.replace_basic_blocks(attention)

    def replace_basic_blocks(self, attention):
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.model, layer_name)
            for index, block in enumerate(layer):
                if isinstance(block, BasicBlock):
                    new_block = ResNetBasicBlockAttention(block, attention)
                    layer[index] = new_block
                    self.attention_blocks.append(new_block.attention)

    def forward(self, x):
        return self.model(x)

    def attention_block(self, index=None):
        if len(self.attention_blocks) == 0:
            return None
        if index is None:
            index = len(self.attention_blocks) // 2
        return self.attention_blocks[index]


def train(model, trainset):
    model.train()
    data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_list = []
    accuracy_list = []
    for epoch in range(epoch_size):
        total_loss, total_correct, total_samples = 0, 0, 0
        time_start = time.time()
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_correct += (torch.argmax(pred, dim=1) == y).sum().item()
            total_samples += len(y)

        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        loss_list.append(epoch_loss)
        accuracy_list.append(epoch_accuracy)
        print(f'epoch:{epoch + 1}, accuracy:{epoch_accuracy:.4f}, loss:{epoch_loss:.4f}, time:{time.time() - time_start:.2f}')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch_size + 1), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curve')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch_size + 1), accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy curve')

    plt.tight_layout()
    plt.show()
    torch.save(model.state_dict(), model_path)


def test(model, testset):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    data_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0, 0, 0
    time_start = time.time()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * len(y)
            total_correct += (torch.argmax(pred, dim=1) == y).sum().item()
            total_samples += len(y)

    print(f'test accuracy:{total_correct / total_samples:.4f}, test loss:{total_loss / total_samples:.4f}, time:{time.time() - time_start:.2f}')


def predict_one(model, testset, sample_index=0):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    image, label = testset[sample_index]
    input_image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_image)
        pred_label = torch.argmax(pred, dim=1).item()

    show_image = denormalize(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(show_image.permute(1, 2, 0))
    plt.title(f'origin: {class_names[label]}\npred: {class_names[pred_label]}')
    plt.axis('off')
    plt.show()


def show_attention(model, testset, sample_indices, attention_index=None):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    attention_block = model.attention_block(attention_index)
    if isinstance(sample_indices, int):
        sample_indices = [sample_indices]

    plt.figure(figsize=(12, 10))
    for row, sample_index in enumerate(sample_indices[:3]):
        image, label = testset[sample_index]
        input_image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_image)
            pred_label = torch.argmax(pred, dim=1).item()

            spatial_attention = attention_block.spatial_attention.last_attention
            channel_attention = attention_block.channel_attention.last_attention

            if spatial_attention is None:
                spatial_attention = torch.ones(1, 1, 224, 224, device=device)
            if channel_attention is None:
                channel_attention = torch.ones(1, 512, 1, 1, device=device)

            spatial_attention = F.interpolate(spatial_attention, size=(224, 224), mode='bilinear', align_corners=False)
            spatial_attention = spatial_attention[0, 0].cpu()
            channel_attention = channel_attention[0, :, 0, 0].cpu()

        plt.subplot(3, 3, row * 3 + 1)
        show_image = denormalize(image)
        plt.imshow(show_image.permute(1, 2, 0))
        plt.title(f'origin: {class_names[label]}\npred: {class_names[pred_label]}')
        plt.axis('off')

        plt.subplot(3, 3, row * 3 + 2)
        plt.imshow(spatial_attention, cmap='jet')
        plt.title('spatial attention')
        plt.axis('off')

        plt.subplot(3, 3, row * 3 + 3)
        plt.bar(range(len(channel_attention)), channel_attention)
        plt.ylim(0, 1)
        plt.xlabel('channel')
        plt.ylabel('weight')
        plt.title('channel attention')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    trainsets, testsets = load_dataset()
    model = ResNet18AttentionNet(attention=attention_type).to(device)
    train(model, trainsets)
    test(model, testsets)
    # show_attention(model, testsets, sample_indices=[0, 10, 20])
    # predict_one(model, testsets, sample_index=0)
