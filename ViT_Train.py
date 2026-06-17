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
from torchvision.models import vit_b_16, ViT_B_16_Weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_root = './data/90'
batch_size = 4
epoch_size = 10
learning_rate = 0.0001
weight_decay = 1e-4
model_path = 'ViT_B16_Rafd90.pth'

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


class ViTAttentionBlock(nn.Module):
    def __init__(self, old_block):
        super().__init__()
        self.ln_1 = old_block.ln_1
        self.self_attention = old_block.self_attention
        self.dropout = old_block.dropout
        self.ln_2 = old_block.ln_2
        self.mlp = old_block.mlp
        self.last_attention = None

    def forward(self, input):
        x = self.ln_1(input)
        x, attention = self.self_attention(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attention = attention.detach()
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class ViTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        self.attention_blocks = nn.ModuleList()
        self.replace_encoder_blocks()

    def replace_encoder_blocks(self):
        for name, block in self.model.encoder.layers.named_children():
            new_block = ViTAttentionBlock(block)
            self.model.encoder.layers[int(name.split('_')[-1])] = new_block
            self.attention_blocks.append(new_block)

    def forward(self, x):
        return self.model(x)

    def attention_block(self, index=-1):
        if len(self.attention_blocks) == 0:
            return None
        return self.attention_blocks[index]


def train(model, trainset):
    model.train()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_list = []
    accuracy_list = []
    for epoch in range(epoch_size):
        total_loss, total_correct, total_samples = 0, 0, 0
        time_start = time.time()
        for x, y in train_loader:
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
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0, 0, 0
    time_start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
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


def get_cls_attention_map(model, image, block_index=-1):
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_image)
        pred_label = torch.argmax(pred, dim=1).item()

    attention_block = model.attention_block(block_index)
    attention = attention_block.last_attention

    # attention: [batch, heads, tokens, tokens]，第0个token是CLS token。
    cls_attention = attention[0, :, 0, 1:].mean(dim=0)
    patch_size = int(cls_attention.numel() ** 0.5)
    cls_attention = cls_attention.reshape(1, 1, patch_size, patch_size)
    cls_attention = F.interpolate(cls_attention, size=(224, 224), mode='bilinear', align_corners=False)
    cls_attention = cls_attention[0, 0].cpu()
    return cls_attention, pred_label


def show_attention(model, testset, sample_indices, block_index=-1):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if isinstance(sample_indices, int):
        sample_indices = [sample_indices]

    plt.figure(figsize=(9, 10))
    for row, sample_index in enumerate(sample_indices[:3]):
        image, label = testset[sample_index]
        attention_map, pred_label = get_cls_attention_map(model, image, block_index)

        plt.subplot(3, 2, row * 2 + 1)
        show_image = denormalize(image)
        plt.imshow(show_image.permute(1, 2, 0))
        plt.title(f'origin: {class_names[label]}\npred: {class_names[pred_label]}')
        plt.axis('off')

        plt.subplot(3, 2, row * 2 + 2)
        plt.imshow(show_image.permute(1, 2, 0))
        plt.imshow(attention_map, cmap='jet', alpha=0.45)
        plt.title(f'ViT CLS attention block {block_index}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    trainsets, testsets = load_dataset()
    model = ViTNet().to(device)
    # train(model, trainsets)
    test(model, testsets)
    # predict_one(model, testsets, sample_index=0)
    show_attention(model, testsets, sample_indices=[0, 10, 20], block_index=-1)
