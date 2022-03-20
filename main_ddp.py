#!/apps/anaconda3/bin/python
import argparse
import copy
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from catalyst.data.sampler import DistributedSamplerWrapper
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed=42, cuda_deterministic=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def pprint(*args):
    # print with current time
    time = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] -"
    if torch.distributed.get_rank() == 0:
        print(time, *args, flush=True)


def _freeze_modules(epoch, model, args):
    if (
        args.freeze_first_n_epochs == 0
        or epoch not in [0, args.freeze_first_n_epochs]
        or args.freeze_modules == ""
    ):
        return

    freeze = True
    if args.freeze_modules[0] == "~":
        freeze = False
        modules = args.freeze_modules[1:].split(",")
    else:
        modules = args.freeze_modules.split(",")

    if epoch == 0:
        pprint("..freeze modules:")
        grad = False
    else:
        pprint("..unfreeze modules:")
        grad = True

    for name, module in model.module.named_children():
        if (freeze and name in modules) or (not freeze and name not in modules):
            print(name, end=", ")
            for param in module.parameters():
                param.requires_grad_(grad)
    print()


def load_state_dict_unsafe(model, state_dict):
    """
    Load state dict to provided model while ignore exceptions.
    """

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model)
    load = None  # break load->load reference cycle

    return {
        "unexpected_keys": unexpected_keys,
        "missing_keys": missing_keys,
        "error_msgs": error_msgs,
    }


# Personalized ImageFolder
class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    # override find_classes
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and entry.name != "love"
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# Sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self, dataset, indices=None, num_samples=None, callback_get_label=None
    ):
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
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


global_step = -1
def train_epoch(epoch, model, optimizer, scheduler, data_loader, writer, args):

    global global_step
    data_loader.sampler.set_epoch(epoch)
    loss_fn = nn.CrossEntropyLoss()

    _freeze_modules(epoch, model, args)
    model.train()

    cnt = 0
    loop = (
        tqdm(data_loader, total=min(args.max_iter_epoch, len(data_loader)))
        if args.local_rank == 0
        else data_loader
    )
    for feat, label in loop:
        if cnt > args.max_iter_epoch:
            break
        cnt += 1

        feat, label = feat.to(args.device), label.to(args.device)

        global_step += 1
        optimizer.zero_grad()

        pred = model(feat)
        loss = loss_fn(pred, label)
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()

        if writer:
            writer.add_scalar("Train/RunningLoss", loss.item(), global_step)

    scheduler.step()

    # save log
    if writer:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(name + ".grad", param.grad, epoch)


def test_epoch(epoch, model, data_loader, writer, args, prefix="Test"):

    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    losses = []
    scores = []

    for feat, label in tqdm(data_loader, desc=prefix, total=len(data_loader)):

        with torch.no_grad():
            feat, label = feat.to(args.device), label.to(args.device)
            pred = model(feat)
            loss = loss_fn(pred, label)
            score = (pred.argmax(dim=1) == label).to(feat).mean()

        losses.append(loss.item())
        scores.append(score.item())

    losses = pd.Series(losses)
    scores = pd.Series(scores)

    if writer:
        writer.add_scalar(prefix + "/Loss/%s" % "CE", losses.mean(), epoch)
        writer.add_scalar(prefix + "/Loss/std(%s)" % "CE", losses.std(), epoch)
        writer.add_scalar(prefix + "/Metric/%s" % "acc", scores.mean(), epoch)
        writer.add_scalar(prefix + "/Metric/std(%s)" % "acc", scores.std(), epoch)

    return losses, scores


def inference(model, data_loader, args, prefix="Test"):

    model.eval()

    preds = []  # Logits: [N * out_feat]
    for feat, label in tqdm(data_loader, desc=prefix, total=len(data_loader)):

        feat, label = feat.to(args.device), label.to(args.device)

        with torch.no_grad():
            pred = model(feat)

        preds.append(
            pd.concat(
                [
                    pd.DataFrame(
                        pred.cpu().numpy(),
                        columns=[c for c in data_loader.dataset.classes],
                    ),
                    pd.DataFrame(
                        {"label": label.cpu().numpy()}, index=np.arange(len(label))
                    ),
                ],
                axis=1,
            )
        )

    preds = pd.concat(preds, axis=0)

    return preds


def main(args):

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    set_seed(args.seed + args.local_rank)

    outdir = args.outdir + "/" + datetime.now().strftime("%m-%d_%H-%M-%S")
    if not os.path.exists(outdir) and args.local_rank == 0:
        os.makedirs(outdir)

    # Transform
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomResizedCrop(224),  # better
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # dataset
    imgF = ImageFolder if args.use_self_imgF else torchvision.datasets.ImageFolder
    train_ds = imgF(args.data_dir + "/train", transform=transform_train)
    test_ds = imgF(args.data_dir + "/test", transform=transform_test)
    args.out_feat = len(train_ds.class_to_idx)
    if args.validate:
        train_ds, valid_ds = torch.utils.data.random_split(
            train_ds, [round(0.9 * len(train_ds)), round(0.1 * len(train_ds))]
        )

    # dataloader
    if args.sampler == "imbalance":
        sampler = DistributedSamplerWrapper(ImbalancedDatasetSampler(train_ds))
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=args.n_workers,
    )

    if args.validate:
        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.n_workers,
        )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.n_workers,
    )

    # model setting
    model = eval("torchvision.models." + args.model)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, args.out_feat)
    if args.init_state:
        pprint("load model init state")
        res = load_state_dict_unsafe(
            model, torch.load(args.init_state, map_location="cpu")
        )
        pprint(res)
    model.to(args.device)
    model = nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=True,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    writer = SummaryWriter(log_dir=outdir) if args.local_rank == 0 else None

    # training
    for epoch in range(args.n_epochs):

        pprint("Epoch:", epoch)

        pprint("training...")
        train_epoch(epoch, model, optimizer, scheduler, train_loader, writer, args=args)

        pprint("evaluating...")
        if args.validate:
            valid_loss, valid_score = test_epoch(
                epoch, model, valid_loader, writer, args=args, prefix="Valid"
            )
        else:
            valid_loss, valid_score = np.array([0]), np.array([0])
        test_loss, test_score = test_epoch(
            epoch, model, test_loader, writer, args=args, prefix="Test"
        )

        pprint(
            "Loss (%s): train %.6f, valid %.6f, test %.6f"
            % ("CE", 0, valid_loss.mean(), test_loss.mean())
        )

        pprint(
            "Metric (%s): train %.6f, valid %.6f, test %.6f"
            % ("Accuracy", 0, valid_score.mean(), test_score.mean())
        )

        if epoch % 4 == 0 and args.local_rank == 0:
            torch.save(model.module.state_dict(), outdir + "/model.bin")

    if args.local_rank == 0:

        # inference
        scores = {}
        names = ["train", "valid", "test"] if args.validate else ["train", "test"]
        for name in names:
            preds = inference(model, eval(name + "_loader"), args, name)
            preds.to_pickle(outdir + f"/{name}_pred.pkl")
            pred = np.argmax(preds.values[:, :-1], axis=1)
            scores[name + "_acc"] = (pred == preds["label"]).astype(float).mean()

        args.device = "cuda:0"
        info = dict(
            config=vars(args),
            scores=scores,
        )

        with open(outdir + "/info.json", "w") as f:
            json.dump(info, f, indent=4)
        pprint("finished.")


def parse_args():

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # model
    parser.add_argument("--out_feat", type=int, default=6)
    parser.add_argument("--init_state", default="")
    parser.add_argument("--model", default="resnet18")

    # training
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--freeze_modules", default="")
    parser.add_argument("--freeze_first_n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=-1)  # -1: no early stop
    parser.add_argument("--loss", default="CE")
    parser.add_argument("--metric", default="acc")
    parser.add_argument("--max_iter_epoch", type=int, default=200)
    parser.add_argument("--milestones", type=int, default=[8, 16, 24], nargs="+")

    # data
    parser.add_argument("--pin_memory", action="store_false")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="random")
    parser.add_argument("--validate", action="store_true")

    # other
    parser.add_argument("--use_self_imgF", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="~/VisualEmotion/data/images")
    parser.add_argument("--outdir", default="./output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_ckpts", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.environ["LOCAL_RANK"]))
    parser.add_argument(
        "--comments", default="", help="add comments without indent and dash`-`"
    )

    args = parser.parse_args()
    args.device = torch.device("cuda", args.local_rank)

    return args


if __name__ == "__main__":

    args = parse_args()
    main(args)
