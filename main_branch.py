import argparse
import os.path
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime
from pointmlp_branch import PointMLP
from dataset import PointDataSet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics


args = argparse.ArgumentParser('training')
args.add_argument('--checkpoint', type=str, default='checkpoints/last_checkpoint.pth')
args.add_argument('--batch_size', type=int, default=16)
args.add_argument('--num_classes', type=int, default=15)
args.add_argument('--dataset', type=str, default='ScanObjectNN')
args.add_argument('--epoch', type=int, default=300)
args.add_argument('--num_points', type=int, default=1024)
args.add_argument('--learning_rate', type=float, default=0.01)
args.add_argument('--weight_decay', type=float, default=1e-4)
args.add_argument('--workers', type=int, default=4)
args = args.parse_args()


def cal_loss(pred, gold):
    gold = gold.contiguous().view(-1) # 将真实的label拉伸成1维
    loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def train(net, train_loader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()

    for batch_idx,(data,label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)  # (batch_size, points_num, 3)
        optimizer.zero_grad()
        logits,features1,features2 = net(data)
        features1 = features1.view(-1)
        features2 = features2.view(-1)
        similarity_loss = torch.cosine_similarity(features1, features2,dim=0)
        loss = criterion(logits, label) + 0.1 * similarity_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]  # 返回最大值以及相应的index

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total = total + label.size(0)
        correct = correct + preds.eq(label).sum().item()

        print("Loss: %.3f | Acc: %.3f%%(%d/%d)" % (train_loss/(batch_idx+1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)

    return {
        "loss": float("%.3f" % (train_loss / (batch_idx+1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true,train_pred))), # 跟上面的acc是否一样
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true,train_pred))),
        "time": time_cost
    }

def validate(net, test_loader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx,(data,label) in enumerate(test_loader):
            data,label = data.to(device),label.to(device)
            logits,features1,features2 = net(data)
            features1 = features1.view(-1)
            features2 = features2.view(-1)
            similarity_loss = torch.cosine_similarity(features1, features2, dim=0)
            loss = criterion(logits, label) + 0.1 * similarity_loss
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx+1), 100. * correct / total, correct, total))
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }

def save_model(net, epoch, acc, is_best, **kwargs):
    state = {
        "net": net.state_dict(),
        'epoch': epoch,
        'acc': acc
    }
    for key,value in kwargs.items():
        state[key] = value
    filepath = "checkpoints/last_checkpoint.pth"
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath,"checkpoints/best_checkpoint.pth")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = PointMLP(points_num=args.num_points, class_num=args.num_classes).to(device)
    criterion = cal_loss # 损失函数
    if device == 'cuda':  # 并行加速训练
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    best_test_acc = 0.
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0
    optimizer_dict = None
    if os.path.isfile(args.checkpoint):  # 如果pth文件，则加载该文件
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        net_dict = checkpoint['net']
        #net_dict = remove_prefix(net_dict, 'module.')
        net.load_state_dict(net_dict)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        optimizer_dict = checkpoint['optimizer']
    train_loader = DataLoader(PointDataSet(num_points=args.num_points, mode='train', data_type=args.dataset),
                              num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(PointDataSet(num_points=args.num_points, mode='test', data_type=args.dataset),
                             num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer,args.epoch,eta_min=args.learning_rate / 100, last_epoch=start_epoch-1)
    # 开始训练
    for epoch in range(start_epoch, args.epoch): # start_epoch, args.epoch
        print('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, train_loader, optimizer, criterion, device)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = max(test_out["acc"], best_test_acc)
        best_train_acc = max(train_out["acc"], best_train_acc)
        best_test_acc_avg = max(test_out["acc_avg"], best_test_acc_avg)
        best_train_acc_avg = max(train_out["acc_avg"], best_train_acc_avg)
        best_test_loss = min(test_out["loss"], best_test_loss)
        best_train_loss = min(train_out["loss"], best_train_loss)

        # 保存模型
        save_model(net,epoch,
                   acc = test_out["acc"],
                   is_best = is_best,
                   best_test_acc = best_test_acc,
                   best_train_acc = best_train_acc,
                   best_test_acc_avg = best_test_acc_avg,
                   best_train_acc_avg = best_train_acc_avg,
                   best_test_loss = best_test_loss,
                   best_train_loss = best_train_loss,
                   optimizer = optimizer.state_dict())

        print(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        print(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")


if __name__ == '__main__':
    main()