import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    googlenet, GoogLeNet_Weights,
    resnet18, ResNet18_Weights
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_root = './data/90'
batch_size = 2
epoch_size = 10
learning_rate = 0.0001
weight_decay = 1e-4

# Select one model: 'googlenet', 'resnet18'
model_name = 'googlenet'

# False: fine-tune all parameters. True: train only the final classifier.
freeze_backbone = False

# Select split: 'person' is recommended for Rafd90 comparison.
split_type = 'person'

do_train = True
do_test = True
do_predict = True

model_path = f'{model_name}_Rafd90_pretrained.pth'

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


def split_indices_random(dataset_size, train_ratio=0.8, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int(train_ratio * dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return train_indices, test_indices


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

    if split_type == 'person':
        train_indices, test_indices = split_indices_by_person(train_dataset)
    else:
        train_indices, test_indices = split_indices_random(len(train_dataset))

    return Subset(train_dataset, train_indices), Subset(test_dataset, test_indices)


def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def create_model(name):
    if name == 'googlenet':
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux1 = None
        model.aux2 = None
        model.aux_logits = False

        if freeze_backbone:
            freeze_all_parameters(model)
            for param in model.fc.parameters():
                param.requires_grad = True
        return model

    if name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if freeze_backbone:
            freeze_all_parameters(model)
            for param in model.fc.parameters():
                param.requires_grad = True
        return model

    raise ValueError("model_name must be 'googlenet' or 'resnet18'")


def load_model_parameters(model):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Cannot find model file: {model_path}. Please train first.')
    model.load_state_dict(torch.load(model_path, map_location=device))


def train(model, trainset):
    model.train()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
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
    load_model_parameters(model)
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
    load_model_parameters(model)
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


if __name__ == '__main__':
    trainsets, testsets = load_dataset()
    model = create_model(model_name).to(device)

    if do_train:
        train(model, trainsets)
    if do_test:
        test(model, testsets)
