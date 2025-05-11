from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("experiments/defence/tokenizer.json")

# Tokenize your text
text = "Your input text here."
output = tokenizer.encode(text)

print("Tokens:", output.tokens)
print("IDs:", output.ids)