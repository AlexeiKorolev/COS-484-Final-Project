from custom_BPE_defence import custom_bpe, corpus_to_vocab
from collections import Counter
import re
import json
from tokenizers import Tokenizer


def apply_merges(word, merges):
    """Tokenize a word using the learned merges."""
    symbols = list(word)
    symbols.append("</w>")
    merges_dict = {"".join(pair): pair for pair in merges}
    while True:
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
        mergeable = [(i, merges_dict.get("".join(pair))) for i, pair in enumerate(pairs) if merges_dict.get("".join(pair))]
        if not mergeable:
            break
        # Merge the first applicable pair
        i, pair = mergeable[0]
        symbols = symbols[:i] + ["".join(pair)] + symbols[i+2:]
    if symbols[-1] == "</w>":
        symbols = symbols[:-1]
    return symbols

def tokenize_corpus(corpus, merges):
    """Tokenize the entire corpus using the learned merges."""
    tokens = []
    words = re.findall(r'\w+', corpus)
    for word in words:
        tokens.extend(apply_merges(word, merges))
    return tokens

def evaluate_tokenization(corpus, merges):
    words = re.findall(r'\w+', corpus)
    total_words = len(words)
    tokens = tokenize_corpus(corpus, merges)
    total_tokens = len(tokens)
    avg_tokens_per_word = total_tokens / total_words if total_words else 0
    print(f"Total words: {total_words}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per word: {avg_tokens_per_word:.3f}")

if __name__ == "__main__":
    real_tokenizer = False
    with open("small_oscar/ru/ru_meta.txt", "r", encoding="utf-8") as f:
        corpus = f.read()[:50000]
    vocab = corpus_to_vocab(corpus)
    initial_vocab_size = len(vocab)

    if real_tokenizer:
        # Load the tokenizer
        tokenizer = Tokenizer.from_file("experiments/defense/tokenizer.json")

        # Tokenize your text
        text = "Your input text here."
        output = tokenizer.encode(corpus)

        words = re.findall(r'\w+', corpus)
        total_words = len(words)

        print("Tokens:", output.tokens)
        print("IDs:", output.ids)

        tokens = output.tokens

        total_tokens = len(tokens)
        avg_tokens_per_word = total_tokens / total_words if total_words else 0
        print(f"Total words: {total_words}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per word: {avg_tokens_per_word:.3f}")

        exit()

    

    # Example corpus

    # num_merges = desired_vocab_size - initial_vocab_size

    # merges, _ = custom_bpe(vocab, num_merges=num_merges, k=10)

    # Load merges from the specified JSON file
    with open("experiments/defence_custom/tokenizer.json", "r", encoding="utf-8") as f:
        merges = json.load(f)["merges"]
    
    print("Evaluating tokenization effectiveness:")
    evaluate_tokenization(corpus, merges)