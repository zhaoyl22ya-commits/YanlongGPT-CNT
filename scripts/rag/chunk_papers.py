from pathlib import Path
import json


# =========================
# 路径
# =========================

BASE_DIR = Path(
    r"E:\YanlongGPT"
)


PAPER_DIR = (
    BASE_DIR
    /
    "data"
    /
    "processed"
    /
    "CNT_clean"
)


SAVE_FILE = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "cnt_chunks.jsonl"
)



SAVE_FILE.parent.mkdir(
    exist_ok=True
)



# =========================
# 参数
# =========================


CHUNK_SIZE = 1000

OVERLAP = 200



# =========================
# chunk函数
# =========================


def split_text(
    text
):


    chunks=[]


    start=0


    length=len(text)



    while start < length:


        end = start + CHUNK_SIZE


        chunk = text[start:end]


        chunks.append(
            chunk
        )


        start = end - OVERLAP



    return chunks





# =========================
# main
# =========================


def main():


    md_files = list(
        PAPER_DIR.glob("*.md")
    )


    print(
        "论文数量:",
        len(md_files)
    )


    total=0



    with open(
        SAVE_FILE,
        "w",
        encoding="utf-8"
    ) as fout:



        for file in md_files:


            print(
                "处理:",
                file.name
            )



            text=file.read_text(
                encoding="utf-8",
                errors="ignore"
            )



            chunks = split_text(
                text
            )



            for i,chunk in enumerate(chunks):


                item={

                    "id":
                    total,


                    "source":
                    file.name,


                    "chunk_id":
                    i,


                    "text":
                    chunk

                }


                fout.write(

                    json.dumps(
                        item,
                        ensure_ascii=False
                    )
                    +
                    "\n"

                )


                total+=1



    print("================")
    print(
        "chunk完成"
    )

    print(
        "总chunk:",
        total
    )



if __name__=="__main__":

    main()