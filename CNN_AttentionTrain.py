import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.models import alexnet, AlexNet_Weights

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.utils.data as Dataloader
from torch.utils.data import Subset


batch_size = 16
epoch_size = 10
attention_type = 'cbam'  # 可选: 'channel', 'spatial', 'cbam'，'none'
model_path = f'AlexNet_Rafd90_{attention_type}_attention.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_root = r'D:\资料\组会\组会2\90'
feature_lr = 1e-5
attention_lr = 1e-4
classifier_lr = 1e-4

class_names = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
num_classes = 8

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
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


def get_person_id(image_path):
    filename = os.path.basename(image_path)
    parts = filename.split('_')
    return parts[1]


def split_by_person_id(samples, test_ratio=0.2):
    person_ids = sorted({get_person_id(path) for path, _ in samples})
    generator = torch.Generator().manual_seed(42)
    person_order = torch.randperm(len(person_ids), generator=generator).tolist()
    test_person_count = max(1, int(len(person_ids) * test_ratio))
    test_person_ids = {person_ids[index] for index in person_order[:test_person_count]}

    train_indices = []
    test_indices = []
    for index, (path, _) in enumerate(samples):
        person_id = get_person_id(path)
        if person_id in test_person_ids:
            test_indices.append(index)
        else:
            train_indices.append(index)

    train_person_ids = set(person_ids) - test_person_ids
    return train_indices, test_indices, train_person_ids, test_person_ids


def create_datasets():
    global class_names, num_classes

    train_full_dataset = ImageFolder(root=dataset_root, transform=train_transform)
    test_full_dataset = ImageFolder(root=dataset_root, transform=test_transform)
    class_names = train_full_dataset.classes
    num_classes = len(class_names)

    train_indices, test_indices, train_person_ids, test_person_ids = split_by_person_id(train_full_dataset.samples)
    train_datasets = Subset(train_full_dataset, train_indices)
    test_datasets = Subset(test_full_dataset, test_indices)

    return train_datasets, test_datasets


def denormalize(image):
    image = image.cpu() * imagenet_std + imagenet_mean
    return image.clamp(0, 1)


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
    def __init__(self, channels, attention='spatial'):
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


class Net(nn.Module):
    def __init__(self, attention='cbam'):
        super().__init__()
        base_model = alexnet(weights=AlexNet_Weights.DEFAULT)

        if attention == 'none':
            self.features = base_model.features
        else:
            self.features = nn.Sequential(
                base_model.features[0],
                base_model.features[1],
                base_model.features[2],
                base_model.features[3],
                base_model.features[4],
                base_model.features[5],
                AttentionBlock(192, attention),
                base_model.features[6],
                base_model.features[7],
                AttentionBlock(384, attention),
                base_model.features[8],
                base_model.features[9],
                AttentionBlock(256, attention),
                base_model.features[10],
                base_model.features[11],
                AttentionBlock(256, attention),
                base_model.features[12],
            )

        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def attention_block(self, index=None):
        blocks = [module for module in self.features if isinstance(module, AttentionBlock)]
        if not blocks:
            return None
        if index is None:
            index = len(blocks) // 2
        return blocks[index]


def create_optimizer(model):
    feature_params = []
    attention_params = []
    classifier_params = list(model.classifier.parameters())

    for module in model.features:
        if isinstance(module, AttentionBlock):
            attention_params.extend(list(module.parameters()))
        else:
            feature_params.extend(list(module.parameters()))

    params = []
    if feature_params:
        params.append({'params': feature_params, 'lr': feature_lr})
    if attention_params:
        params.append({'params': attention_params, 'lr': attention_lr})
    if classifier_params:
        params.append({'params': classifier_params, 'lr': classifier_lr})

    return optim.Adam(params, weight_decay=1e-5)


def train(model, train_datasets):
    model.train()
    dataloader = Dataloader.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)
    loss_list = []
    accuracy_list = []

    for epoch in range(epoch_size):
        total_loss, total_correct, total_samples = 0, 0, 0
        time_start = time.time()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pre = model(x)
            loss = loss_func(y_pre, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_correct += (torch.argmax(y_pre, dim=1) == y).sum().item()
            total_samples += len(y)

        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        loss_list.append(epoch_loss)
        accuracy_list.append(epoch_accuracy)
        print(f'epoch:{epoch + 1}, correct_rate: {epoch_accuracy:.2f}, loss: {epoch_loss:.5f}, time: {time.time() - time_start}')

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


def test(model, test_datasets):
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    loss_func = nn.CrossEntropyLoss()
    dataloader = Dataloader.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)
    total_loss, total_correct, total_samples = 0, 0, 0
    time_start = time.time()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pre = model(x)
            loss = loss_func(y_pre, y)
            total_loss += loss.item() * len(y)
            total_correct += (torch.argmax(y_pre, dim=1) == y).sum().item()
            total_samples += len(y)

    print(f'accuracy: {total_correct / total_samples:.2f}, loss: {total_loss / total_samples:.5f}, time: {time.time() - time_start}')


def show_attention(model, test_datasets, sample_indices, attention_index=None):
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    if isinstance(sample_indices, int):
        sample_indices = [sample_indices]

    plt.figure(figsize=(12, 10))

    for row, sample_index in enumerate(sample_indices[:3]):
        image, label = test_datasets[sample_index]
        input_image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            y_pre = model(input_image)
            pred_label = torch.argmax(y_pre, dim=1).item()
            attention_block = model.attention_block(attention_index)
            if attention_block is None:
                spatial_attention = torch.ones(1, 1, 224, 224, device=device)
                channel_attention = torch.ones(1, 256, 1, 1, device=device)
            else:
                spatial_attention = attention_block.spatial_attention.last_attention
                channel_attention = attention_block.channel_attention.last_attention

                if spatial_attention is None:
                    spatial_attention = torch.ones(1, 1, 224, 224, device=device)
                if channel_attention is None:
                    channel_attention = torch.ones(1, 256, 1, 1, device=device)

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
        plt.title('middle spatial attention')
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
    train_datasets, test_datasets = create_datasets()
    model = Net(attention=attention_type).to(device)
    train(model, train_datasets)
    test(model, test_datasets)
    # show_attention(model, test_datasets, sample_indices=[52, 520, 5200], attention_index=0)
