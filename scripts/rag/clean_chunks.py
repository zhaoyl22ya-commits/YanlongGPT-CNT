import json
import re
from pathlib import Path


# ===============================
# 路径设置
# ===============================

# 你的原始chunk文件
INPUT_FILE = Path(
    r"E:\YanlongGPT\data\rag\cnt_chunks.jsonl"
)


# 清洗后输出
OUTPUT_FILE = Path(
    r"E:\YanlongGPT\data\chunks\clean_chunks.jsonl"
)



# ===============================
# 清洗函数
# ===============================

def clean_text(text):

    # -----------------------
    # 1. 删除Page标记
    # -----------------------
    text = re.sub(
        r"# Page \d+",
        "",
        text
    )


    # -----------------------
    # 2. 删除DOI行
    # -----------------------
    text = re.sub(
        r"DOI:.*",
        "",
        text
    )


    # -----------------------
    # 3. 删除下载信息
    # -----------------------
    text = re.sub(
        r"Downloaded from.*",
        "",
        text
    )


    # -----------------------
    # 4. 删除Article标记
    # -----------------------
    text = re.sub(
        r"\bArticle\b",
        "",
        text
    )


    # -----------------------
    # 5. 删除作者信息
    # -----------------------

    text = re.split(
        r"■AUTHOR INFORMATION",
        text
    )[0]


    # -----------------------
    # 6. 删除参考文献
    # -----------------------

    text = re.split(
        r"■REFERENCES",
        text
    )[0]


    # -----------------------
    # 7. 删除Acknowledgment
    # -----------------------

    text = re.split(
        r"■ACKNOWLEDGMENTS",
        text
    )[0]


    # -----------------------
    # 8. 删除空白
    # -----------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    # 去前后空格

    return text.strip()



# ===============================
# 主程序
# ===============================


def main():

    print("开始清洗chunk...")


    count = 0
    removed = 0


    OUTPUT_FILE.parent.mkdir(
        exist_ok=True
    )


    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as fin, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as fout:


        for line in fin:


            item = json.loads(line)


            old_text = item["text"]


            new_text = clean_text(
                old_text
            )


            # 空chunk删除

            if len(new_text) < 50:
                removed += 1
                continue


            item["text"] = new_text


            fout.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )


            count += 1


            if count % 5000 == 0:
                print(
                    f"已处理 {count} chunks"
                )


    print("===================")

    print(
        "清洗完成"
    )

    print(
        f"保留chunk: {count}"
    )

    print(
        f"删除chunk: {removed}"
    )

    print(
        "输出:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()