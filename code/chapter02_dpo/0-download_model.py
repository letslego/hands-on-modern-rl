"""
2 0： ModelScope 
====================================

， Qwen2.5-0.5B-Instruct 。
，。

：
    pip install modelscope
    python 0-download_model.py
"""

import os
from modelscope import snapshot_download

# 
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"

# ModelScope  ID
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def download_model():
    if os.path.exists(LOCAL_MODEL_DIR) and os.path.exists(
        os.path.join(LOCAL_MODEL_DIR, "config.json")
    ):
        print(f" {LOCAL_MODEL_DIR}，。")
        print(f"， {LOCAL_MODEL_DIR} 。")
        return LOCAL_MODEL_DIR

    print(f" ModelScope  {MODEL_ID} ...")
    print(" 1GB，。")
    #  local_dir  cache_dir：cache_dir  repo 
    # （./Qwen2.5-0.5B-Instruct/Qwen/Qwen2___5-0___5B-Instruct/），
    #  LOCAL_MODEL_DIR  config.json。
    model_dir = snapshot_download(
        MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print(f"，：{model_dir}")
    return model_dir


if __name__ == "__main__":
    download_model()
