from datasets import load_dataset


options = ['af', 'am', 'an', 'ar', 'arz', 'as', 'ast', 'av', 'az', 'azb', 'ba', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs', 'bxr', 'ca', 'ce', 'ceb', 'ckb', 'cs', 'cv', 'cy', 'da', 'de', 'dsb', 'dv', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gn', 'gom', 'gsw', 'gu', 'he', 'hi', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ie', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'krc', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lez', 'li', 'lmo', 'lo', 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms', 'mt', 'multi', 'mwl', 'my', 'mzn', 'nah', 'nds', 'ne', 'new', 'nl', 'nn', 'no', 'oc', 'or', 'os', 'pa', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'ro', 'ru', 'sa', 'sah', 'sd', 'sh', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'vo', 'wa', 'war', 'wuu', 'x-eml', 'xal', 'xmf', 'yi', 'yo', 'zh']

# Download the OSCAR-2301 dataset
def download_oscar_dataset(option):
    dataset = load_dataset("oscar-corpus/OSCAR-2301", option, split="all")
    return dataset

if __name__ == "__main__":
    for option in options:
        oscar_dataset = download_oscar_dataset(option)
        oscar_dataset.save_to_disk(f"./oscar_dataset/{option}")
        print("Dataset downloaded and saved successfully!")