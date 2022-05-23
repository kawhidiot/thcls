import os
from sklearn.metrics import classification_report, confusion_matrix
import torch
from collections import OrderedDict
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from model import ClsModel


def main():
    batch_size = 24
    nw = 1
    img_size = 256
    src_path = r'E:\0Ecode\code220328_cls\dataset\data2'
    model_name = 'mobilenetv2_100'

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

    val_dir = os.path.join(src_path, 'test')
    validate_dataset = datasets.ImageFolder(root=val_dir,
                                            transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    net = ClsModel(model_name, len(validate_dataset.classes))
    net.to(device)

    save_path = './weights/{}.pth'.format(model_name)

    state_dict = torch.load(save_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(state_dict)

    pred_all = []
    gt_all = []
    with torch.no_grad():
        for step, val_data in enumerate(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            pred = list(torch.argmax(outputs, 1).detach().cpu().numpy())
            gt = list(val_labels.detach().cpu().numpy())
            pred_all += pred
            gt_all += gt
    print('************eval result*********')
    print(classification_report(gt_all, pred_all))
    print(confusion_matrix(gt_all, pred_all))




if __name__ == '__main__':
    main()
