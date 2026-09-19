import json
from pathlib import Path
import sys

import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer



# =====================================================
# 路径
# =====================================================


BASE_DIR = Path(
    r"E:\YanlongGPT"
)


# YanlongGPT代码路径

sys.path.append(
    str(BASE_DIR / "scripts")
)



# =====================================================
# 导入你的模型
# =====================================================


from yanlongllm import (
    YanlongLLM,
    YanlongConfig
)



# =====================================================
# 文件路径
# =====================================================


MODEL_FILE = (

    BASE_DIR
    /
    "checkpoint_sft"
    /
    "yanlongllm_cnt_sft_epoch3.pt"

)


TOKENIZER_FILE = (

    BASE_DIR
    /
    "tokenizer"
    /
    "cnt_tokenizer.json"

)


BGE_PATH = (

    BASE_DIR
    /
    "models"
    /
    "bge-base-en-v1.5"

)



INDEX_FILE = (

    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
    /
    "cnt_faiss.index"

)



METADATA_FILE = (

    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
    /
    "metadata.json"

)



CHUNK_FILE = (

    BASE_DIR
    /
    "data"
    /
    "chunks"
    /
    "clean_chunks.jsonl"

)



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


from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file(

    str(TOKENIZER_FILE)

)



eos_id = tokenizer.token_to_id(
    "[EOS]"
)



# =====================================================
# 加载BGE
# =====================================================


print(
    "加载BGE..."
)


embed_model = SentenceTransformer(

    str(BGE_PATH),

    device=device

)



# =====================================================
# 加载FAISS
# =====================================================


print(
    "加载FAISS..."
)


index = faiss.read_index(

    str(INDEX_FILE)

)



print(
    "FAISS:",
    index.ntotal
)



# =====================================================
# metadata
# =====================================================


with open(

    METADATA_FILE,

    "r",

    encoding="utf-8"

) as f:


    metadata = json.load(f)



# =====================================================
# chunk文本
# =====================================================


chunk_texts={}



with open(

    CHUNK_FILE,

    "r",

    encoding="utf-8"

) as f:


    for line in f:


        item=json.loads(line)


        chunk_texts[item["id"]] = item["text"]




print(
    "chunk加载完成:",
    len(chunk_texts)
)



# =====================================================
# 加载YanlongGPT
# =====================================================


print(
    "加载YanlongGPT..."
)



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



checkpoint=torch.load(

    MODEL_FILE,

    map_location=device

)



model.load_state_dict(

    checkpoint["model"]

)



model.to(device)

model.eval()



print(
    "YanlongGPT加载完成"
)



# =====================================================
# 检索
# =====================================================


def retrieve(query, top_k=5):


    vector = embed_model.encode(

        [query],

        normalize_embeddings=True

    )


    vector=np.array(

        vector,

        dtype=np.float32

    )



    scores, ids=index.search(

        vector,

        top_k

    )



    contexts=[]



    for idx in ids[0]:


        meta=metadata[idx]


        text=chunk_texts.get(

            meta["id"],

            ""

        )


        contexts.append(text)



    return contexts




# =====================================================
# 生成
# =====================================================


@torch.no_grad()
def generate(question):


    contexts = retrieve(

        question,

        top_k=5

    )


    context="\n\n".join(

        contexts

    )



    prompt=(

        "### Context:\n"

        +

        context

        +

        "\n\n### Instruction:\n"

        +

        question

        +

        "\n\n### Response:\n"

    )



    ids=tokenizer.encode(

        prompt

    ).ids



    input_ids=torch.tensor(

        ids,

        dtype=torch.long

    ).unsqueeze(0).to(device)



    for _ in range(200):


        logits,_=model(

            input_ids[:,-512:]

        )


        logits=logits[:,-1,:]


        next_token=torch.argmax(

            logits,

            dim=-1,

            keepdim=True

        )


        input_ids=torch.cat(

            [

                input_ids,

                next_token

            ],

            dim=1

        )


        if eos_id:

            if next_token.item()==eos_id:

                break



    output=input_ids[0].tolist()[len(ids):]


    return tokenizer.decode(output), contexts




# =====================================================
# chat
# =====================================================


if __name__=="__main__":


    print("===================")

    print("YanlongGPT-RAG")

    print("输入exit退出")

    print("===================")



    while True:


        q=input("\nUser:\n>")


        if q=="exit":

            break



        answer,ctx=generate(q)



        print("\nRetrieved context:")

        print(ctx[0][:500])


        print("\nYanlongGPT:")

        print(answer)