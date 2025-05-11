import random
from collections import Counter, defaultdict

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in v_in:
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = v_in[word]
    return v_out

def custom_bpe(vocab, num_merges, k=10):
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        if (i+1) % (k+1) == 0:
            # Random merge
            pair = random.choice(list(pairs.keys()))
        else:
            # Standard BPE: most frequent pair
            pair = max(pairs, key=pairs.get)
        merges.append(pair)
        vocab = merge_vocab(pair, vocab)
    return merges, vocab


# Example usage:
# vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, ...}
# merges, final_vocab = custom_bpe(vocab, num_merges=100, k=10)

def corpus_to_vocab(corpus):
    """
    Converts a list of sentences into a vocab dictionary suitable for BPE.
    Each word is split into characters and ends with </w>.
    """
    vocab = Counter()
    for line in corpus.split("\n"):
        words = line.split(" ")
        for word in words:
            chars = (" ".join(list(word))) + "</w>"
            vocab[chars] += 1
    return dict(vocab)


# Example usage:
# corpus = ["low lower", "newest lowest"]
# vocab = corpus_to_vocab(corpus)