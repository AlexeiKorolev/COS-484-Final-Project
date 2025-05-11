from custom_BPE_defence import custom_bpe, corpus_to_vocab
import argparse
import os
import json
from collections import Counter, defaultdict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer with defense mechanism.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the tokenizer files will be saved.")

    parser.add_argument("--corpus_dir", type=str, required=True, help="Path to the directory containing the corpus files.")
    args = parser.parse_args()



    corpus = "hi"

    for root, dirs, files in os.walk(args.corpus_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    corpus += f.read()
    
    

    vocab = corpus_to_vocab(corpus)

    

    merges, vocab = custom_bpe(vocab, num_merges=1000, k=999999)

    tokenizer_data = {
        "merges": merges,
        "vocab": vocab
    }

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")

    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)

    print(f"Tokenizer saved to {tokenizer_path}")
