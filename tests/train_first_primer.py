import numpy as np

with open('firstprimer.txt', 'r') as f:
    text = f.read()

# 1. Create the Character Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 2. Create Mappings (String to Int, Int to String)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# 3. Tokenize the entire book
data = np.array([stoi[c] for c in text], dtype=np.uint32)

print(f"Dataset length: {len(data)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")

# 4. Export to a C Header
with open('train_first_primer.h', 'w') as f:
    f.write(f"#ifndef PRIMER_DATA_H\n#define PRIMER_DATA_H\n\n")
    f.write(f"#define VOCAB_SIZE {vocab_size}\n")
    f.write(f"#define DATA_LEN {len(data)}\n\n")
    
    # Export the decoding array so your C engine can "speak" later
    f.write("const char itos_map[] = {")
    char_literals = []
    for c in chars:
        if c == '\n': char_literals.append("'\\n'")
        elif c == '\\': char_literals.append("'\\\\'")
        elif c == '\'': char_literals.append("'\\''")
        else: char_literals.append(f"'{c}'")
    f.write(", ".join(char_literals))
    f.write("};\n\n")
    
    # Export the actual training data
    f.write("const unsigned int train_data[] = {\n")
    f.write(", ".join(map(str, data)))
    f.write("\n};\n\n#endif\n")
