import torch
from pathlib import Path
from torch.utils.data import DataLoader
import sys


# =====================================================
# 路径
# =====================================================

BASE_DIR = Path(r"E:\YanlongGPT")


CHECKPOINT_FILE = (
    BASE_DIR
    /
    "checkpoint"
    /
    "yanlongllm_cnt_epoch5.pt"
)


SAVE_DIR = (
    BASE_DIR
    /
    "checkpoint_sft"
)

SAVE_DIR.mkdir(
    exist_ok=True
)


# =====================================================
# 导入模型
# =====================================================


sys.path.append(
    str(BASE_DIR / "scripts")
)

from yanlongllm import (
    YanlongLLM,
    YanlongConfig
)


from scripts.sft_dataset import CNT_SFT_Dataset



# =====================================================
# 参数
# =====================================================

batch_size = 4

block_size = 512

learning_rate = 2e-5

epochs = 5



device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "device:",
    device
)



# =====================================================
# 数据
# =====================================================


dataset = CNT_SFT_Dataset(
    max_length=block_size
)


train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)



print(
    "SFT样本:",
    len(dataset)
)



# =====================================================
# 模型
# =====================================================


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



# =====================================================
# 加载预训练模型
# =====================================================


print(
    "加载base模型:"
)

print(
    CHECKPOINT_FILE
)


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)


model.load_state_dict(
    checkpoint["model"]
)


print(
    "模型加载完成"
)



model.to(device)



print(
    "参数量:",
    sum(
        p.numel()
        for p in model.parameters()
    )/1e6,
    "M"
)



# =====================================================
# optimizer
# =====================================================


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=learning_rate

)



# =====================================================
# SFT训练
# =====================================================


for epoch in range(epochs):


    model.train()


    total_loss = 0



    for step,batch in enumerate(train_loader):


        input_ids = batch["input_ids"].to(device)


        labels = batch["labels"].to(device)



        logits,loss = model(

            input_ids,

            labels

        )



        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



        if step % 50 == 0:


            print(

                f"epoch {epoch+1}",

                f"step {step}",

                f"loss {loss.item():.4f}"

            )



    avg_loss = (
        total_loss
        /
        len(train_loader)
    )



    print("====================")


    print(
        "epoch:",
        epoch+1
    )


    print(
        "SFT loss:",
        avg_loss
    )



    # 保存

    torch.save(

        {

            "model":
            model.state_dict(),


            "config":
            config.__dict__,


            "epoch":
            epoch+1

        },


        SAVE_DIR
        /
        f"yanlongllm_cnt_sft_epoch{epoch+1}.pt"

    )



print("====================")

print(
    "SFT训练完成"
)

print(
    SAVE_DIR
)