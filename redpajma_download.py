from datasets import load_dataset
import os

TARGET_BYTES = 5 * 1024 * 1024 * 1024  # 5GB

def download_5gb_redpajama(split="train", subset=None):
    # If you want only code, set subset="code"
    if subset:
        ds = load_dataset("togethercomputer/RedPajama-Data-1T", subset, split=split, streaming=True)
        out_dir = f"./redpajama_dataset/{subset}"
    else:
        ds = load_dataset("togethercomputer/RedPajama-Data-1T", split=split, streaming=True)
        out_dir = "./oscar_dataset/code"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "code_meta.txt")
    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            # RedPajama-1T uses "text" field
            text = sample.get("text")
            if not text:
                continue
            encoded = text + "\n"
            encoded_bytes = encoded.encode("utf-8")
            if total_bytes + len(encoded_bytes) > TARGET_BYTES:
                remaining = TARGET_BYTES - total_bytes
                f.write(encoded_bytes[:remaining].decode("utf-8", errors="ignore"))
                break
            f.write(encoded)
            total_bytes += len(encoded_bytes)
    print(f"Downloaded {total_bytes / (1024*1024*1024):.2f} GB to {out_path}")

if __name__ == "__main__":
    # For all data:
    # download_5gb_redpajama()
    # For code only:
    download_5gb_redpajama(subset="github")