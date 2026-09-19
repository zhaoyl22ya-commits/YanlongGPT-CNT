import json
from pathlib import Path

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


# BGE模型

MODEL_PATH = (
    BASE_DIR
    /
    "models"
    /
    "bge-base-en-v1.5"
)


# FAISS索引

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


# metadata

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
# 加载BGE
# =====================================================


print(
    "加载embedding模型..."
)


embed_model = SentenceTransformer(

    str(MODEL_PATH),

    device=device

)


embed_model.eval()


print(
    "embedding模型加载完成"
)



# =====================================================
# 加载FAISS
# =====================================================


print(
    "加载FAISS索引..."
)


index = faiss.read_index(

    str(INDEX_FILE)

)


print(

    "FAISS向量数量:",

    index.ntotal

)



# =====================================================
# 加载metadata
# =====================================================


with open(

    METADATA_FILE,

    "r",

    encoding="utf-8"

) as f:

    metadata = json.load(f)



print(
    "metadata加载完成"
)



# =====================================================
# 检索函数
# =====================================================


def retrieve(

    query,

    top_k=5

):


    """
    输入:
        query:
            用户问题

        top_k:
            返回多少个相关chunk


    输出:
        最相关论文片段
    """



    # -------------------------
    # 1. query embedding
    # -------------------------


    query_vector = embed_model.encode(

        [query],

        normalize_embeddings=True

    )


    query_vector = np.array(

        query_vector,

        dtype=np.float32

    )



    # -------------------------
    # 2. FAISS搜索
    # -------------------------


    scores, ids = index.search(

        query_vector,

        top_k

    )



    results=[]



    # -------------------------
    # 3. 找回原文信息
    # -------------------------


    for score, idx in zip(

        scores[0],

        ids[0]

    ):


        if idx == -1:

            continue



        item = metadata[idx]


        results.append(

            {

                "score":

                float(score),


                "id":

                item["id"],


                "source":

                item["source"],


                "chunk_id":

                item["chunk_id"]

            }

        )



    return results





# =====================================================
# 测试
# =====================================================


if __name__ == "__main__":


    while True:


        query=input(

            "\nQuestion:\n> "

        )


        if query=="exit":

            break



        results = retrieve(

            query,

            top_k=5

        )



        print(

            "\n========== Retrieved =========="

        )


        for i,r in enumerate(results):


            print(

                f"\nRank {i+1}"

            )


            print(

                "score:",

                r["score"]

            )


            print(

                "source:",

                r["source"]

            )


            print(

                "chunk:",

                r["chunk_id"]

            )


        print(

            "=============================="

        )