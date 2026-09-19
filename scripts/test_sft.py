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


print(
    "device:",
    device
)



# =====================================================
# tokenizer
# =====================================================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)


print(
    "词表大小:",
    tokenizer.get_vocab_size()
)


eos_id = tokenizer.token_to_id(
    "[EOS]"
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



print(
    "SFT模型加载完成"
)



# =====================================================
# generate
# =====================================================


@torch.no_grad()
def generate(
    question,
    max_new_tokens=200,
    temperature=0.3
):


    # 必须和训练格式一致

    prompt = (

        "### Instruction:\n"

        +

        question

        +

        "\n\n### Response:\n"

    )



    prompt_ids = tokenizer.encode(
        prompt
    ).ids



    input_ids = torch.tensor(

        prompt_ids,

        dtype=torch.long

    ).unsqueeze(0)



    input_ids = input_ids.to(device)



    for _ in range(max_new_tokens):


        idx = input_ids[:, -512:]


        logits, _ = model(
            idx
        )


        logits = logits[:, -1, :]



        # =================================================
        # repetition penalty
        # =================================================

        generated_tokens = set(
            input_ids[0].tolist()
        )


        for token_id in generated_tokens:

            logits[0, token_id] /= 1.2



        # =================================================
        # top-k sampling
        # =================================================


        top_k = 10


        values, indices = torch.topk(
            logits,
            top_k
        )



        values = values / temperature



        probs = torch.softmax(
            values,
            dim=-1
        )



        choice = torch.multinomial(
            probs,
            num_samples=1
        )



        next_token = indices.gather(
            -1,
            choice
        )



        input_ids = torch.cat(

            [
                input_ids,
                next_token
            ],

            dim=1

        )



        # EOS停止

        if eos_id is not None:

            if next_token.item() == eos_id:

                break




    # 去掉prompt

    generated_ids = (

        input_ids[0]

        .tolist()

        [
            len(prompt_ids):
        ]

    )



    answer = tokenizer.decode(
        generated_ids
    )


    return answer





# =====================================================
# 测试
# =====================================================


if __name__ == "__main__":


    questions = [


        "How does catalyst particle size affect carbon nanotube growth?",


        "How does temperature influence CNT structure and yield during CVD growth?",


        "Why are Fe catalysts commonly used for carbon nanotube growth?",


        "How can the G peak and D peak in Raman spectroscopy be used to evaluate CNT quality?",


        "How can CNT diameter be controlled through experimental conditions?"

    ]



    for q in questions:


        print(
            "=============================="
        )


        print(
            "Question:"
        )


        print(q)



        print(
            "\nAnswer:"
        )


        answer = generate(

            q,

            max_new_tokens=200,

            temperature=0.3

        )


        print(answer)