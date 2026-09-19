import json
from pathlib import Path

import torch
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer



# =====================================================
# 路径
# =====================================================


BASE_DIR = Path(
    r"E:\YanlongGPT"
)



# 清洗后的chunk文件

INPUT_FILE = (
    BASE_DIR
    /
    "data"
    /
    "chunks"
    /
    "clean_chunks.jsonl"
)



# embedding保存位置

OUTPUT_DIR = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
)


OUTPUT_DIR.mkdir(
    exist_ok=True
)



# =====================================================
# 参数
# =====================================================


# 本地BGE模型

MODEL_NAME = (
    r"E:\YanlongGPT\models\bge-base-en-v1.5"
)



# RTX4060 8GB建议

BATCH_SIZE = 16



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
# 加载embedding模型
# =====================================================


print(
    "加载embedding模型..."
)


model = SentenceTransformer(

    MODEL_NAME,

    device=device

)


model.eval()


print(
    "模型加载完成"
)



# =====================================================
# 读取chunk
# =====================================================


texts = []

metadata = []


print(
    "读取chunks..."
)



with open(

    INPUT_FILE,

    "r",

    encoding="utf-8"

) as f:


    for idx, line in enumerate(f):


        item = json.loads(line)



        # 检查字段

        if "text" not in item:

            continue



        texts.append(

            item["text"]

        )


        metadata.append(

            {

                "id":
                item.get(
                    "id",
                    idx
                ),


                "source":
                item.get(
                    "source",
                    ""
                ),


                "chunk_id":
                item.get(
                    "chunk_id",
                    idx
                )

            }

        )




print(
    "chunk数量:",
    len(texts)
)



# =====================================================
# embedding
# =====================================================


print(
    "开始embedding..."
)



embeddings = model.encode(

    texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    normalize_embeddings=True,

    convert_to_numpy=True

)



# 确保float32

embeddings = embeddings.astype(

    np.float32

)



print(

    "embedding shape:",

    embeddings.shape

)



# =====================================================
# 保存embedding
# =====================================================



embedding_file = (

    OUTPUT_DIR

    /

    "embeddings.npy"

)



np.save(

    embedding_file,

    embeddings

)



# =====================================================
# 保存metadata
# =====================================================



metadata_file = (

    OUTPUT_DIR

    /

    "metadata.json"

)



with open(

    metadata_file,

    "w",

    encoding="utf-8"

) as f:


    json.dump(

        metadata,

        f,

        ensure_ascii=False,

        indent=2

    )




print("==============================")

print(
    "embedding完成"
)


print(
    "向量文件:",
    embedding_file
)


print(
    "索引信息:",
    metadata_file
)


print("==============================")
''









































































































































































































































































































































































































































































































































































