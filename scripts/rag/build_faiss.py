import numpy as np
import json
import faiss

from pathlib import Path



# =====================================================
# 路径
# =====================================================


BASE_DIR = Path(
    r"E:\YanlongGPT"
)


EMBED_DIR = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
)



EMBED_FILE = (
    EMBED_DIR
    /
    "embeddings.npy"
)


METADATA_FILE = (
    EMBED_DIR
    /
    "metadata.json"
)



INDEX_FILE = (
    EMBED_DIR
    /
    "cnt_faiss.index"
)



# =====================================================
# 加载embedding
# =====================================================


print("加载embedding...")


embeddings = np.load(
    EMBED_FILE
)


print(
    "embedding shape:",
    embeddings.shape
)



# =====================================================
# 建立FAISS索引
# =====================================================


dimension = embeddings.shape[1]


print(
    "向量维度:",
    dimension
)



# 因为已经normalize
# 使用内积 = cosine similarity

index = faiss.IndexFlatIP(
    dimension
)



print(
    "加入向量..."
)


index.add(
    embeddings
)



print(
    "向量数量:",
    index.ntotal
)



# =====================================================
# 保存index
# =====================================================


faiss.write_index(
    index,
    str(INDEX_FILE)
)



print("===================")

print(
    "FAISS建立完成"
)


print(
    INDEX_FILE
)