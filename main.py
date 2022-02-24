import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import Callable
import copy
import json
import argparse
from tqdm import tqdm
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]



def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

# Sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    
# train
global_step = -1
def train_epoch(epoch, model, optimizer, scheduler, data_loader, writer, args):

    global global_step
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for feat, label in tqdm(data_loader, total=len(data_loader)):
        
        feat, label = feat.to(device), label.to(device)

        global_step += 1
        optimizer.zero_grad()
        
        pred = model(feat)
        loss = loss_fn(pred, label)
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
        
        if writer:
            writer.add_scalar('Train/RunningLoss', loss.item(), global_step)

    scheduler.step()
    
    # save log
    if writer:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(name+'.grad', param.grad, epoch)

                
def test_epoch(epoch, model, data_loader, writer, args, prefix='Test'):

    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    losses = []
    scores = []

    for feat, label in tqdm(data_loader, desc=prefix, total=len(data_loader)):

        feat, label=feat.to(device), label.to(device)
        with torch.no_grad():
            pred = model(feat)

        loss = loss_fn(pred, label)
        score = (pred.argmax(dim=1)==label).to(feat).mean()
        
        losses.append(loss.item())
        scores.append(score.item())

    losses = pd.Series(losses)
    scores = pd.Series(scores) 
    
    if writer:
        writer.add_scalar(prefix+'/Loss/%s'%"CE", losses.mean(), epoch)
        writer.add_scalar(prefix+'/Loss/std(%s)'%"CE", losses.std(), epoch)
        writer.add_scalar(prefix+'/Metric/%s'%"acc", scores.mean(), epoch)
        writer.add_scalar(prefix+'/Metric/std(%s)'%"acc", scores.std(), epoch)

    return losses, scores


def inference(model, data_loader, args, prefix='Test'):

    model.eval()

    preds = [] # Logits: [N * 7]
    for feat, label in tqdm(data_loader, desc=prefix, total=len(data_loader)):

        feat, label = feat.to(device), label.to(device)

        with torch.no_grad():
            pred = model(feat)

        preds.append(pd.concat([
            pd.DataFrame(pred.cpu().numpy(), columns=[c for c in data_loader.dataset.classes]),
            pd.DataFrame({'label': label.cpu().numpy()}, index=np.arange(len(label)))
        ],axis=1))

    preds = pd.concat(preds, axis=0)

    return preds


def main(args):

    set_seed()

    # Transform
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(40),
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010])])

    # dataset 
    train_ds = torchvision.datasets.ImageFolder(args.data_dir + "/images/train", transform=transform_train)
    valid_ds = torchvision.datasets.ImageFolder(args.data_dir + "/images/valid", transform=transform_train)
    test_ds = torchvision.datasets.ImageFolder(args.data_dir + "/images/test", transform=transform_test)

    # dataloader
    batch_size=256
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(train_ds), drop_last=True, num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, 
        sampler=ImbalancedDatasetSampler(valid_ds), drop_last=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, 
        sampler=ImbalancedDatasetSampler(test_ds), drop_last=True, num_workers=0)


    # model setting
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.to(device)
    # model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_epochs//5, gamma=args.gamma)

    # loss
    writer = SummaryWriter(log_dir="./log")
    best_score = {}
    best_epoch = 0
    stop_round = 0
    best_param = copy.deepcopy(model.state_dict())

    for epoch in range(args.n_epochs):
        stop_round += 1
        if stop_round > args.early_stop and args.early_stop >= 0:
            print('early stop')
            break

        print('Epoch:', epoch)

        print('training...')
        train_epoch(epoch, model, optimizer, scheduler, train_loader, writer, args=args)

        print('evaluating...')
        # train_loss, train_score = test_epoch(
        #     epoch, model, train_loader, writer=None, args=None, prefix='Train')
        valid_loss, valid_score = test_epoch(
            epoch, model, valid_loader, writer, args=args, prefix='Valid')
        test_loss, test_score = test_epoch(
            epoch, model, test_loader, writer, args=args, prefix='Test')

        print('Loss (%s): train %.6f, valid %.6f, test %.6f'%(
            "CE", 0, valid_loss.mean(), test_loss.mean()))
    
        print('Metric (%s): train %.6f, valid %.6f, test %.6f'%(
            "Accuracy", 0, valid_score.mean(), test_score.mean()))


        best_valid_score = best_score.get('valid_'+ args.metric, -1)
        if valid_score.mean() > best_valid_score:
            print('\tvalid metric (%s) updates from %.6f to %.6f'%(
                args.metric, best_valid_score, valid_score.mean()))                
            for name in ['valid', 'test']:
                best_score[name+'_loss'] = eval(name+'_loss').mean()
                best_score[name+'_'+args.metric] = eval(name+'_score').mean()
            stop_round = 0
            best_epoch = epoch    
            best_param = copy.deepcopy(model.state_dict())
            torch.save(best_param, f'./model/model.bin')
    
    
    print('best score:', best_score, '@', best_epoch)
    info = dict(
        config={"early_stop":args.early_stop, "n_epochs": args.n_epochs},
        best_epoch=best_epoch,
        best_score=best_score,
    )
    with open('./model/info.json', 'w') as f:
        json.dump(info, f, indent=4)

    # inference
    model.load_state_dict(torch.load(args.init_state, map_location='cpu'))
    for name in ['train','valid','test']:
        preds = inference(model, eval(name+"_loader"), None, name)
        preds.to_pickle(args.outdir+name+"_pred.pkl")
    print('finished.')
    


def parse_args():

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # model
    parser.add_argument('--d_feat', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_drop', action='store_true')
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--init_state', default='./model/model.bin')

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.9)    
    parser.add_argument('--weight_decay', type=float, default=0.9)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--loss', default='CE')
    parser.add_argument('--metric', default='acc')

    # data
    parser.add_argument('--pin_memory', action='store_false')
    parser.add_argument('--batch_size', type=int, default=256) # -1 indicate daily batch

    # other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', default='~/VisualEmotion/data')
    parser.add_argument('--outdir', default='./')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save_ckpts', action='store_true')

    args = parser.parse_args()

    return args



if __name__ == '__main__':

    args = parse_args()
    main(args)



