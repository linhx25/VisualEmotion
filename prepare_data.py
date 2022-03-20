#!/apps/anaconda3/bin/python
import io
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pandas as pd
import requests
import shutil
import glob
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torchvision
import re
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
split = 0.9
main_dir = "images"
model_dir = "./output/03-19_12-08-23/model.bin"


def label_dicts():
    """Returns dictionaries mapping level-3 labels to levels 1 and 2, respectively."""

    # Index of labels corresponds to label column in
    # dataframe e.g., 'affection' is '0' in train/test
    labels = [
        ("affection", "love", "+"),
        ("cheerfullness", "joy", "+"),
        ("confusion", "sadness", "-"),
        ("contentment", "joy", "+"),
        ("disappointment", "sadness", "-"),
        ("disgust", "anger", "-"),
        ("enthrallment", "joy", "+"),
        ("envy", "anger", "-"),
        ("exasperation", "anger", "-"),
        ("gratitude", "love", "+"),
        ("horror", "fear", "-"),
        ("irritabilty", "anger", "-"),
        ("lust", "love", "+"),
        ("neglect", "sadness", "-"),
        ("nervousness", "fear", "-"),
        ("optimism", "joy", "+"),
        ("pride", "joy", "+"),
        ("rage", "anger", "-"),
        ("relief", "joy", "+"),
        ("sadness", "sadness", "-"),
        ("shame", "sadness", "-"),
        ("suffering", "sadness", "-"),
        ("surprise", "surprise", "+"),
        ("sympathy", "sadness", "-"),
        ("zest", "joy", "+"),
    ]

    lvl_one = {}
    lvl_two = {}
    lvl_three = {}
    for idx, val in enumerate(labels):
        lvl_one[idx] = val[2]
        lvl_two[idx] = val[1]
        lvl_three[idx] = val[0]

    return lvl_one, lvl_two, lvl_three


def load_data():

    ## Load train/test WEBEmo URLs and labels and shuffle
    df = pd.concat(
        [
            pd.read_csv(
                r"~/VisualEmotion/data/WEBEmo/train25.txt", sep=" ", header=None
            ).rename(columns={0: "url", 1: "label"}),
            pd.read_csv(
                r"~/VisualEmotion/data/WEBEmo/test25.txt", sep=" ", header=None
            ).rename(columns={0: "url", 1: "label"}),
        ]
    ).sample(frac=1)

    ## split train/test
    n_split = int(len(df) * split)
    df["index"] = np.arange(len(df))
    df["split"] = df.apply(
        lambda x: "train" if x["index"] < n_split else "test", axis=1
    )

    ## generate data dir
    _, lvl_2, lvl_3 = label_dicts()
    df["root_dir"] = df.apply(
        lambda x: f"./data/{main_dir}/"
        + x["split"]
        + "/"
        + lvl_2[x["label"]]
        + "/"
        + lvl_3[x["label"]],
        axis=1,
    )

    ## move disgust
    df.loc[df.label == 5, "root_dir"] = df.loc[df.label == 5, "root_dir"].apply(
        lambda x: x.replace("anger/disgust", "disgust")
    )

    ## mkdirs
    dirs = df["root_dir"].unique()
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    return df


def save_imgs(df):
    """Retrieves an image from its URL and saves locally.
    Args:
        df (dataframe): Pandas dataframe containing image
          URLs and associated labels.
    Returns:
        None. Image is saved locally.
    """
    response = requests.get(df[1]["url"])
    img = Image.open(io.BytesIO(response.content))
    img_name = df[1]["url"].split("/")[-1]
    label = str(df[1]["label"])
    path = os.path.join(df[1]["root_dir"], label + "_" + img_name)
    img.save(path)
    return


def pool_image_retrieval(df):
    """Utilizes multithreading to call `save_imgs()` function.
    Args:
        df (dataframe): Pandas dataframe containing image
          URLs and associated labels.
    Returns:
        None. Images are saved locally.
    """
    pool = ThreadPool(os.cpu_count())
    pool.map(func=save_imgs, iterable=df.iterrows())
    return


def get_WEBEmo():
    df = load_data()
    pool_image_retrieval(df)


def get_UnbiasedEmo():
    for path in glob.glob("./data/UnBiasedEmo/images/*"):
        klass = path.split("/")[-1]
        imgs = glob.glob(path + "/**/*.jpg", recursive=True)
        # imgs = glob.glob(path+'/*/*.jpg')
        random.shuffle(imgs)
        n = len(imgs)
        n_split = int(n * split)
        print(f"class: {klass}, all: {n}, train: {n_split}, test: {n-n_split}")

        for i, img in enumerate(tqdm(imgs)):
            if i < n_split:
                tgt = img.replace("UnBiasedEmo/images", f"{main_dir}/train")
            else:
                tgt = img.replace("UnBiasedEmo/images", f"{main_dir}/test")

            if not os.path.exists("/".join(tgt.split("/")[:-1])):
                os.makedirs("/".join(tgt.split("/")[:-1]))

            # print(img, tgt)
            if os.path.exists(img):
                shutil.move(img, tgt)


def get_Emotion6():
    if os.path.exists("./data/Emotion-6/images/anger/digust"):  # typo in their dataset
        shutil.move(
            "./data/Emotion-6/images/anger/digust", "./data/Emotion-6/images/disgust"
        )
    for path in glob.glob("./data/Emotion-6/images/*"):
        klass = path.split("/")[-1]
        imgs = glob.glob(path + "/**/*.jpg", recursive=True)
        random.shuffle(imgs)
        n = len(imgs)
        n_split = int(n * split)
        print(f"class: {klass}, all: {n}, train: {n_split}, test: {n-n_split}")

        for i, img in enumerate(tqdm(imgs)):
            if i < n_split:
                tgt = img.replace("Emotion-6/images", f"{main_dir}/train")
            else:
                tgt = img.replace("Emotion-6/images", f"{main_dir}/test")

            if not os.path.exists("/".join(tgt.split("/")[:-1])):
                os.makedirs("/".join(tgt.split("/")[:-1]))

            # print(img, tgt)
            if os.path.exists(img):
                shutil.move(img, tgt)


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


def inference(model, imgF, path=f"./data/{main_dir}"):

    model.eval()
    # dataloader
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

    preds = []  # Logits: [N * 6]
    for name in ["train", "test"]:
        loader = imgF(path + f"/{name}/love", transform=transform_test)
        loader = torch.utils.data.DataLoader(
            loader, batch_size=256, drop_last=False, num_workers=0
        )
        indices = [it[0] for it in loader.dataset.imgs]

        t = []
        for feat, _ in tqdm(loader, desc=name, total=len(loader)):

            feat = feat.to(device)

            with torch.no_grad():
                pred = model(feat).softmax(dim=1)

            t.append(
                pd.DataFrame(
                    pred.cpu().numpy(),
                    columns=["anger", "disgust", "fear", "joy", "sadness", "surprise"],
                )
            )

        t = pd.concat(t, axis=0)
        t.index = indices
        preds.append(t)
    preds = pd.concat(preds, axis=0)

    return preds


def relabel_love(init_state=model_dir):
    # model
    model_name = "resnet50"
    model = eval("torchvision.models." + model_name)(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    res = load_state_dict_unsafe(model, torch.load(init_state, map_location="cpu"))
    model.to(device)
    print(res)

    pred_love = inference(model, torchvision.datasets.ImageFolder)
    joy = pred_love[pred_love.idxmax(axis=1) == "joy"]
    neutral = pred_love[~pred_love.index.isin(joy.index)]
    neutral = ((neutral - 1 / 6) ** 2).sum(axis=1).sort_values()
    topn = len(neutral[neutral < neutral.quantile(0.3)])
    neutral = neutral.head(topn)

    return neutral, joy


def move_relabel(move=False, move_list=""):

    if move_list and move:
        move_list = pd.read_csv(move_list, index_col=0)
        for _, (src, tgt) in tqdm(move_list.iterrows()):
            if os.path.exists(src):
                shutil.move(src, tgt)
        return

    neutral, joy = relabel_love()
    neutral = neutral.index.tolist()
    joy = joy.index.tolist()

    move_list = []

    # move to neutral
    for name in ["train", "test"]:
        if not os.path.exists(f"./data/{main_dir}/{name}/neutral"):
            os.makedirs(f"./data/{main_dir}/{name}/neutral")

    for i, img in enumerate(tqdm(neutral)):
        split = re.search(f"{main_dir}/([a-zA-Z]+)/love", img)[0].split("/")[1]
        src = img
        if split == "train":
            tgt = img.replace("train/love", "train/neutral/love")
        elif split == "test":
            tgt = img.replace("test/love", "test/neutral/love")
        if not os.path.exists("/".join(tgt.split("/")[:-1])):
            os.makedirs("/".join(tgt.split("/")[:-1]))

        move_list.append((src, tgt))
        if os.path.exists(src) and move:
            shutil.move(src, tgt)

    # move to joy
    for i, img in enumerate(tqdm(joy)):
        split = re.search(f"{main_dir}/([a-zA-Z]+)/love", img)[0].split("/")[1]
        src = img
        if split == "train":
            tgt = img.replace("train/love", "train/joy/love")
        elif split == "test":
            tgt = img.replace("test/love", "test/joy/love")
        if not os.path.exists("/".join(tgt.split("/")[:-1])):
            os.makedirs("/".join(tgt.split("/")[:-1]))

        move_list.append((src, tgt))
        if os.path.exists(src) and move:
            shutil.move(src, tgt)

    move_list = pd.DataFrame(move_list, columns=["src", "tgt"])
    model_name = model_dir.split("/")[-2]
    move_list.to_csv(f"./data/moveList_{model_name}.csv")


def show_image():

    from IPython.display import display, Image

    model_name = model_dir.split("/")[-2]
    move_list = pd.read_csv(f"./data/moveList_{model_name}.csv", index_col=0)
    move_list = move_list.sample(frac=1)
    cnt = 0
    for _, (src, tgt) in tqdm(move_list.iterrows()):
        if cnt > 8:
            break
        klass = tgt.split("/")[4]
        if klass == "neutral":
            cnt += 1
            display(Image(filename=src))


if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    ## WEBEmo
    get_WEBEmo()

    ## UnbiasedEmo
    get_UnbiasedEmo()

    ## Emotion-6
    get_Emotion6()

    ## Inference on Love, relabel and move
    move_relabel(move=True)
