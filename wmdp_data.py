# 创建文件：download_wmdp_data.py
import os
from datasets import load_dataset

def download_wmdp_data():
    """下载WMDP数据集到本地目录"""
    
    # 创建目录
    os.makedirs("data/wmdp/wmdp-corpora", exist_ok=True)
    
    # 下载数据集
    datasets_to_download = [
        ("cais/wmdp-bio-forget-corpus", "bio-forget-corpus.jsonl"),
        # ("cais/wmdp-bio-retain-corpus", "bio-retain-corpus.jsonl"), 
        # ("cais/wmdp-cyber-forget-corpus", "cyber-forget-corpus.jsonl"),
        # ("cais/wmdp-cyber-retain-corpus", "cyber-retain-corpus.jsonl"),
    ]
    
    for repo_id, filename in datasets_to_download:
        print(f"Downloading {repo_id}...")
        try:
            # 从HF下载数据集
            ds = load_dataset(repo_id)
            
            # 保存为jsonl格式
            output_path = f"data/wmdp/wmdp-corpora/{filename}"
            ds['train'].to_json(output_path)
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error downloading {repo_id}: {e}")

if __name__ == "__main__":
    download_wmdp_data()