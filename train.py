import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from model import ClsModel
from tensorboardX import SummaryWriter


def vis_info(info, model_name):
    title_names = 'train_accurate, train_loss_avg, val_accurate, val_loss_avg'.split(', ')
    info = np.asarray(info)
    for i in range(len(title_names)):
        col = list(info[:, i])
        plt.plot([k for k in range(len(col))], col)
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.title(title_names[i])
        plt.savefig(os.path.join('curves', '{}_{}.png').format(model_name, title_names[i]))
        plt.clf()


def main():
    batch_size = 24
    nw = 1
    img_size = 256
    src_path = r'E:\0Ecode\code220328_cls\dataset\data2'
    model_name = 'mobilenetv2_100'

    writer = SummaryWriter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("| Device:", device)

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((img_size, img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 加载训练集和验证集
    train_dir = os.path.join(src_path, 'train')
    assert os.path.exists(train_dir), "{} path does not exist.".format(train_dir)
    val_dir = os.path.join(src_path, 'test')
    assert os.path.exists(train_dir), "{} path does not exist.".format(val_dir)

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=val_dir,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("| Train dir:", train_dir)
    print("| Train num:", train_num)
    print("| Val dir:", val_dir)
    print("| Val num:", val_num)
    print("| Bath size:", batch_size)
    print("| Number of workers:", nw)

    net = ClsModel(model_name, len(train_dataset.classes))
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 20
    best_acc = 0.0
    save_path = './weights/{}.pth'.format(model_name)
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    print("training.....")
    start_time = time.time()
    notes = []

    for epoch in range(epochs):
        # train
        net.train()
        train_loss, train_acc, cur_acc, cur_loss = 0.0, 0.0, 0.0, 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            cur_loss = loss.item()
            train_loss += cur_loss
            predict_y = torch.max(logits, dim=1)[1]
            cur_acc = torch.eq(predict_y, labels.to(device)).sum().item()
            train_acc += cur_acc
            print('\rcurrent epoch:{}, train_progress:{}/{}, loss:{:5.3}, acc:{:5.3}'.format(epoch, step*batch_size, train_num, cur_loss, cur_acc*1.00/batch_size), end='')

        train_accurate = train_acc / train_num
        train_loss_avg = train_loss / train_steps
        writer.add_scalar('train_loss_avg', train_loss_avg, epoch)
        writer.add_scalar('train_accurate', train_accurate, epoch)

        # validate
        net.eval()
        val_loss, val_acc, cur_acc, cur_loss = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for step, val_data in enumerate(validate_loader):
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                cur_loss = loss.item()
                val_loss += cur_loss
                predict_y = torch.max(outputs, dim=1)[1]
                cur_acc = torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_acc += cur_acc
                print('\rcurrent epoch:{}, val_progress:{}/{}, loss:{:5.3}, acc:{:5.3}'.format(epoch, step*batch_size, val_num, cur_loss, cur_acc*1.00/batch_size), end='')

        val_accurate = val_acc / val_num
        val_loss_avg = val_loss / val_steps
        writer.add_scalar('val_loss_avg', val_loss_avg, epoch)
        writer.add_scalar('val_accurate', val_accurate, epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        print(
            '\nEpoch:{:3d}, train accuracuy:{:5.3f}, train loss:{:5.3f},  val accuracy:{:5.3f}, val loss:{:5.3f}\n'.format(
                epoch + 1, train_accurate, train_loss_avg, val_accurate, val_loss_avg))
        notes.append([train_accurate, train_loss_avg, val_accurate, val_loss_avg])

    vis_info(notes, model_name)
    end_time = time.time()
    print("| Time cost:", end_time - start_time)
    print('Done.')


if __name__ == '__main__':
    main()
