import os
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sys


sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from yanlongllm import YanlongLLM, YanlongConfig


# =========================
# 路径
# =========================

BASE_DIR = Path(r"E:\YanlongGPT")


TRAIN_FILE = (
    BASE_DIR
    /
    "data"
    /
    "train"
    /
    "train.bin"
)


VAL_FILE = (
    BASE_DIR
    /
    "data"
    /
    "train"
    /
    "val.bin"
)


SAVE_DIR = (
    BASE_DIR
    /
    "checkpoint"
)

SAVE_DIR.mkdir(
    exist_ok=True
)


# =========================
# 超参数
# =========================

batch_size = 8

block_size = 512

learning_rate = 3e-4

epochs = 5


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("device:", device)


# =========================
# Dataset
# =========================

class TokenDataset(Dataset):

    def __init__(
        self,
        file,
        block_size
    ):

        self.data = np.memmap(
            file,
            dtype=np.uint16,
            mode="r"
        )

        self.block_size = block_size



    def __len__(self):

        # 非重叠切块
        return (
            len(self.data)
            -
            self.block_size
        ) // self.block_size



    def __getitem__(self, idx):

        # 每次移动512 token
        start = idx * self.block_size


        x = torch.tensor(
            self.data[
                start:
                start+self.block_size
            ],
            dtype=torch.long
        )


        y = torch.tensor(
            self.data[
                start+1:
                start+self.block_size+1
            ],
            dtype=torch.long
        )


        return x,y



train_dataset = TokenDataset(
    TRAIN_FILE,
    block_size
)


val_dataset = TokenDataset(
    VAL_FILE,
    block_size
)


print(
    "训练样本:",
    len(train_dataset)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)



val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size
)



# =========================
# 模型
# =========================

config = YanlongConfig(
    vocab_size=8000,
    block_size=512,
    n_layer=6,
    n_head=8,
    n_embd=384
)


model = YanlongLLM(
    config
)


model.to(device)



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)



print(
    "参数量:",
    sum(
        p.numel()
        for p in model.parameters()
    )/1e6,
    "M"
)



# =========================
# 验证
# =========================

@torch.no_grad()
def evaluate():

    model.eval()

    losses=[]


    for x,y in val_loader:

        x=x.to(device)
        y=y.to(device)


        _,loss=model(
            x,y
        )


        losses.append(
            loss.item()
        )


    model.train()


    return sum(losses)/len(losses)



# =========================
# Training
# =========================


for epoch in range(epochs):


    model.train()


    total_loss=0


    for step,(x,y) in enumerate(train_loader):


        x=x.to(device)
        y=y.to(device)



        logits,loss=model(
            x,
            y
        )


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



        if step % 100 == 0:

            print(
                f"epoch {epoch+1}",
                f"step {step}",
                f"loss {loss.item():.4f}"
            )


        # 每1000步保存一次
        if step % 1000 == 0 and step != 0:

            torch.save(
                {
                    "model":model.state_dict(),
                    "config":config.__dict__,
                    "epoch":epoch,
                    "step":step
                },
                SAVE_DIR /
                "yanlongllm_cnt_latest.pt"
            )



    val_loss=evaluate()


    print("====================")


    print(
        "epoch:",
        epoch+1
    )


    print(
        "train loss:",
        total_loss/len(train_loader)
    )


    print(
        "val loss:",
        val_loss
    )


    torch.save(
        {
            "model":model.state_dict(),
            "config":config.__dict__
        },
        SAVE_DIR /
        f"yanlongllm_cnt_epoch{epoch+1}.pt"
    )


print("训练完成")