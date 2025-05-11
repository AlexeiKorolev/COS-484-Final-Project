import json
with open(r"experiments\defence_custom\tokenizer.json", encoding="utf-8") as f:
    data = json.load(f)
merges = data["merges"]
with open("experiments/defence_custom/merges.txt", "w", encoding="utf-8") as out:
    for merge in merges:
        out.write((merge[0] + " " + merge[1] + "\n").replace("</w>", ""))