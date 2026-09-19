import torch
from pathlib import Path
import sys


# =====================================================
# 路径
# =====================================================

BASE_DIR = Path(
    r"E:\YanlongGPT"
)


MODEL_FILE = (
    BASE_DIR
    /
    "checkpoint_sft"
    /
    "yanlongllm_cnt_sft_epoch5.pt"
)


TOKENIZER_FILE = (
    BASE_DIR
    /
    "tokenizer"
    /
    "cnt_tokenizer.json"
)



# =====================================================
# import
# =====================================================

sys.path.append(
    str(BASE_DIR / "scripts")
)


from yanlongllm import (
    YanlongLLM,
    YanlongConfig
)


from tokenizers import Tokenizer



# =====================================================
# device
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("device:", device)



# =====================================================
# tokenizer
# =====================================================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)


eos_id = tokenizer.token_to_id(
    "[EOS]"
)


print(
    "vocab:",
    tokenizer.get_vocab_size()
)

print(
    "EOS:",
    eos_id
)



# =====================================================
# model
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



checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)


model.load_state_dict(
    checkpoint["model"]
)


model.to(device)

model.eval()


print("==============================")
print("YanlongGPT CNT SFT 已加载")
print("输入 exit 退出")
print("==============================")



# =====================================================
# generate
# =====================================================


@torch.no_grad()
def generate(
        question,
        max_new_tokens=256,
        temperature=0.7
):


    prompt = (

        "### Instruction:\n"

        +

        question

        +

        "\n\n### Response:\n"

    )


    ids = tokenizer.encode(
        prompt
    ).ids



    input_ids = torch.tensor(
        ids,
        dtype=torch.long
    ).unsqueeze(0)


    input_ids = input_ids.to(device)



    for _ in range(max_new_tokens):


        idx = input_ids[:, -512:]


        logits, _ = model(
            idx
        )


        logits = logits[:, -1, :]



        # =====================
        # sampling
        # =====================


        logits = logits / temperature


        probs = torch.softmax(
            logits,
            dim=-1
        )


        next_token = torch.multinomial(
            probs,
            num_samples=1
        )



        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )



        if eos_id is not None:

            if next_token.item() == eos_id:

                break



    output_ids = (
        input_ids[0]
        .tolist()
        [
            len(ids):
        ]
    )


    answer = tokenizer.decode(
        output_ids
    )


    return answer



# =====================================================
# chat loop
# =====================================================


while True:


    print("\nUser:")

    question = input("> ")



    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print(
            "Bye."
        )

        break



    answer = generate(
        question
    )


    print("\nYanlongGPT:")

    print(answer)
