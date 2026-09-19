# YanlongGPT-CNT

A domain-specific language model and RAG-enhanced scientific assistant
for carbon nanotube (CNT) synthesis research.

## Overview

YanlongGPT-CNT is a self-built Transformer-based language model designed
for scientific knowledge understanding in carbon nanotube synthesis.

The project implements an AI4Science workflow:

    CNT scientific literature
              |
              v
    Data preprocessing
              |
              v
    Domain language model pretraining
              |
              v
    Supervised fine-tuning (SFT)
              |
              v
    Scientific knowledge retrieval (RAG)
              |
              v
    CNT research assistant

## Main Features

-   Self-built decoder-only Transformer language model
-   CNT-domain corpus pretraining
-   Supervised fine-tuning using CNT scientific QA data
-   BGE embedding based semantic retrieval
-   FAISS vector database
-   Retrieval-augmented generation

## Model Information

-   Model type: Decoder-only Transformer
-   Vocabulary size: 8000 tokens
-   Domain: Carbon nanotube synthesis and processing
-   Fine-tuned model: `yanlongllm_cnt_sft.pt`

## RAG Knowledge System

The retrieval pipeline includes:

-   Scientific document chunking
-   Text cleaning
-   Embedding generation
-   FAISS indexing
-   Context retrieval
-   RAG-enhanced generation

Knowledge base statistics:

-   CNT scientific chunks: 56,146
-   Embedding dimension: 768

## Repository Structure

    YanlongGPT-CNT/

    ├── model/
    │   └── yanlongllm_cnt_sft.pt
    │
    ├── tokenizer/
    │   └── cnt_tokenizer.json
    │
    ├── scripts/
    │   ├── yanlongllm.py
    │   ├── chat_sft.py
    │   ├── train/
    │   └── rag/
    │
    ├── data/
    ├── requirements.txt
    ├── environment.yml
    └── README.md

## Installation

### Conda

``` bash
conda env create -f environment.yml
conda activate yanlonggpt-cnt
```

### Pip

``` bash
pip install -r requirements.txt
```

## Running

### SFT inference

``` bash
python scripts/chat_sft.py
```

### RAG assistant

Prepare your own scientific corpus and run:

``` bash
python scripts/rag/build_embedding.py
python scripts/rag/build_faiss.py
python scripts/rag/chat_rag.py
```

## Example Questions

    What catalyst is suitable for high density CNT forest growth?

    How does catalyst particle size affect CNT diameter?

    How does temperature influence CNT morphology during CVD growth?

## Data Availability

Original scientific papers are not included due to copyright
restrictions.

Users can provide their own scientific corpus and run the preprocessing
pipeline.

## License

MIT License
