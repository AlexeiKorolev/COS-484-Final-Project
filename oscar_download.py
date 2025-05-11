from datasets import load_dataset
import os

# List only the most relevant languages you want
relevant_languages = ['en', 'fr', 'es', 'de', 'zh']  # Example: English, French, Spanish, German, Chinese
cyrilic_languages = ['ru', 'uk', 'bg', 'sr']  # Added Russian, Ukrainian, Bulgarian, Serbian
east_asian_languages = ['ja', 'ko']  # Japanese, Korean, Chinese (already have that)
south_asian_languages = ['hi', 'bn', 'ta', 'te', 'ur']  # Hindi, Bengali, Tamil, Telugu, Urdu

languages_to_download = ['en', 'ru', 'zh']# ['ja'] # ['ja', 'ta','te'] # ['zh'] #cyrilic_languages + east_asian_languages + south_asian_languages

TARGET_BYTES = int(0.1 * 1024 * 1024 * 1024)  # 1GB

def download_1gb_oscar(option):
    dataset = load_dataset("oscar-corpus/OSCAR-2301", option, split="train", streaming=True)
    os.makedirs(f"./small_oscar/{option}", exist_ok=True)
    out_path = f"./small_oscar/{option}/{option}_meta.txt"
    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            text = sample.get("text") or sample.get("content")
            if not text:
                continue
            encoded = text + "\n"
            encoded_bytes = encoded.encode("utf-8")
            if total_bytes + len(encoded_bytes) > TARGET_BYTES:
                # Only write up to the target size
                remaining = TARGET_BYTES - total_bytes
                f.write(encoded_bytes[:remaining].decode("utf-8", errors="ignore"))
                break
            f.write(encoded)
            total_bytes += len(encoded_bytes)
    print(f"Downloaded {total_bytes / (1024*1024):.2f} MB for {option}")

if __name__ == "__main__":
    for option in languages_to_download:
        download_1gb_oscar(option)