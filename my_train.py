import timeit
from datetime import datetime
import socket
import glob
import os
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model

# 1 使用GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("Device being used:", device)

# 2 相关超参数配置
nEpochs = 201        # 迭代次数
resume_epoch = 0     # 重复实验
useTest = True       # 验证集使用
nTestInterval = 20   # 多久用一次验证集
snapshot = 5        # 多久保存一次数据
lr = 1e-5

# 3 选择文件
dataset = 'ucf101'

if dataset == 'ucf101':
    num_classes = 101
elif dataset == 'hdmb51':
    num_classes = 51
else:
    print('We only have hmdb and ucf.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

"""
    run_id 保存
    source ： AI人工智能初学者
"""
if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D'
saveName = modelName + '-' + dataset

def train_model(dataset = dataset, save_dir = save_dir, num_classes = num_classes, lr = lr,
                num_epochs = nEpochs, save_epoch = snapshot, useTest = useTest, test_interval = nTestInterval):
    # 1 导入模型
    model = C3D_model.C3D (num_classes = num_classes, pretrained = False)
    train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                    {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]

    # 2 损失函数
    criterion = nn.CrossEntropyLoss()
    # 3 优化函数
    optimizer = optim.SGD(train_params, lr = lr, momentum = 0.9, weight_decay = 5e-4)
    # 4 学习率下降
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model.cuda()
    criterion.cuda()

    # 6 设置训练集和测试机
    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset = dataset, split = 'train', clip_len = 16), batch_size = 4, shuffle = True, num_workers = 0)
    val_dataloader   = DataLoader(VideoDataset(dataset = dataset, split = 'val', clip_len = 16), batch_size = 4, num_workers = 0)
    test_dataloader  = DataLoader(VideoDataset(dataset = dataset, split = 'test', clip_len = 16), batch_size = 4, num_workers = 0)

    trainval_loaders = {'train':train_dataloader, 'val':val_dataloader}
    trainval_sizes = {x:len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    # 8 开始训练模型
    for epoch in range (0, num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                # 根据训练集还是测试集进行模型输出选择
                if phase == 'train':
                    outputs = model(inputs)
                else:
                    # 禁用梯度下降上下文管理器
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim = 1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss  = criterion(outputs, labels.long())

                # 训练集，进行反向传播
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()* inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc  = running_corrects.double() / trainval_sizes[phase]

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        # 保存模型
        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        # 走测试集
        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.cuda()
                labels = labels.cuda()

                with torch.no_grad():
                    outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss  = criterion(outputs, labels.long())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size


            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

if __name__ == "__main__":
    train_model()

