import torch
import torch.nn as nn
import time
import torch.optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

batch_size = 64

transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
datasets = datasets.ImageFolder(root='./data/90', transform=transform)


def LoadDatasets():
    train_size = int(0.8 * len(datasets))
    eval_size = len(datasets) - train_size
    train_datasets, eval_datasets = random_split(datasets, [train_size, eval_size])
    return train_datasets, eval_datasets


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 采用block5容易过拟合，全连接层参数量过多
        # 参数量过大，显存不够
        """
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        """

        self.FC = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(256 * 7 * 7, 4096),    # 删除块5，降低参数量到千万级
            # nn.ReLU(),
            nn.Linear(256 * 7 * 7, 512),  # 进一步减小参数量
            nn.ReLU(),
            nn.Linear(512, 8)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        x = self.FC(x)
        return x


def train(model, train_datasets):
    model.train()
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_list = []
    accuracy_list = []
    epoch = 10
    for epoch_index in range(epoch):
        total_correct, total_samples, total_loss = 0, 0, 0
        time_start = time.time()
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*len(y)
            total_correct += (torch.argmax(pred, dim=1) == y).sum().item()
            total_samples += len(y)
        loss_list.append(total_loss / total_samples)
        accuracy_list.append(total_correct / total_samples)
        print(f'epoch:{epoch_index + 1}, accurracy:{total_correct / total_samples}:.2f, loss:{total_loss / len(train_datasets)}, time:{time.time() - time_start}')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curve')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy curve')

    plt.tight_layout()
    plt.show()
    torch.save(model.state_dict(), 'MyVGGNet_test1.pth')


def evaluate(model, test_datasets):
    model.eval()
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=False)
    model.load_state_dict(torch.load('MyVGGNet_test1.pth'))
    loss_func = nn.CrossEntropyLoss()
    total_correct, total_samples, total_loss = 0, 0, 0
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        loss = loss_func(pred, y)
        total_correct += (torch.argmax(pred, dim=1) == y).sum().item()
        total_samples += len(y)
        total_loss += loss.item()*len(y)

    # 对图像表情进行预测
    img, label = test_datasets[23]
    x = img.unsqueeze(0).cuda()
    with torch.no_grad():
        y_pre = model(x)
        y = torch.argmax(y_pre, dim=1).item()
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f'predict:{y}, label:{label}')
    plt.axis('off')
    plt.show()
    print(f'predict:{y}, label:{label}')
    print(f'accuracy:{total_correct / total_samples}:.2f, loss:{total_loss / len(test_datasets)}')


if __name__ == '__main__':
    train_datasets, eval_datasets = LoadDatasets()
    model = VGGNet().cuda()
    # train(model, train_datasets)
    evaluate(model, train_datasets)
    """
    print(f'类别名称：', datasets.classes)
    print(f'类别编号：', datasets.class_to_idx)
    print(f'train_size, test_size', len(train_datasets), len(eval_datasets))
    """

