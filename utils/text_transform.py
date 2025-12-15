import ast

vietchar_path = r"D:\Dowloads_1\S2T\Vietnamese_char.txt"
with open(vietchar_path, "r", encoding="utf-8") as f:
    content = f.read()
vietchar = ast.literal_eval(f"[{content}]")


class TextTransform:
    def __init__(self):
        spec_char = ["<BLANK>"]
        all_char = spec_char + vietchar
        self.idx2char = {i: char for i, char in enumerate(all_char)}
        self.char2idx = {char: i for i, char in enumerate(all_char)}
        self.all_char = all_char

    def text2int(self, text):
        return [self.char2idx[c] for c in text.lower() if c in self.char2idx]

    def int2text(self, label):
        return "".join([self.idx2char[i] for i in label])

    def get_vocab_size(self):
        return len(self.idx2char)


text_transform = TextTransform()
