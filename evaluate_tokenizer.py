import argparse
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np





def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def boundary_vector(tokens, text):
    """Return a binary vector indicating token boundaries in the original text."""
    vector = [0] * len(text)
    pos = 0
    for token in tokens:
        pos = text.find(token, pos)
        if pos == -1:
            continue  # Skip if token not found
        vector[pos] = 1
        pos += len(token)
    return vector

def evaluate(corpus, tokenizer, reference_tokenizer):
    precisions, recalls, f1s = [], [], []

    for i, line in enumerate(corpus):
        if i % (len(corpus) // 100 + 1) == 0 or i == len(corpus) - 1:
            progress = (i + 1) / len(corpus) * 100
            print(f"\rProgress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}% - Processing line {i + 1}/{len(corpus)}", end='')
        ref_tokens = reference_tokenizer(line)
        pred_tokens = tokenizer(line)

        ref_bv = boundary_vector(ref_tokens, line)
        pred_bv = boundary_vector(pred_tokens, line)

        # Pad shorter one with zeros
        max_len = max(len(ref_bv), len(pred_bv))
        ref_bv += [0] * (max_len - len(ref_bv))
        pred_bv += [0] * (max_len - len(pred_bv))

        precisions.append(precision_score(ref_bv, pred_bv, zero_division=0))
        recalls.append(recall_score(ref_bv, pred_bv, zero_division=0))
        f1s.append(f1_score(ref_bv, pred_bv, zero_division=0))

    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }

# === Example custom tokenizer ===
def custom_tokenizer(text):
    return text.split()  # Replace with your tokenizer

# === Main Script ===
if __name__ == '__main__':
        

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to .txt corpus')
    args = parser.parse_args()
    print("Reading corpus...")
    corpus = read_corpus(args.file)
    print(f"Corpus loaded with {len(corpus)} lines.")
    corpus = read_corpus(args.file)
    results = evaluate(corpus, custom_tokenizer, custom_tokenizer)

    print("Evaluation Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
