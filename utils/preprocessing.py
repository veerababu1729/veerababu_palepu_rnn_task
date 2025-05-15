import os
import numpy as np

def load_classification_data(path):
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            text, label = parts
            texts.append(text.lower())
            labels.append(label)
    return texts, labels

def build_vocab(texts):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def texts_to_sequences(texts, vocab):
    return [[vocab.get(w, vocab['<UNK>']) for w in text.split()] for text in texts]

def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded[i, :length] = seq[:length]
    return padded

def encode_labels(labels):
    label_map = {'Math': 0, 'Science': 1, 'History': 2}
    return np.array([label_map.get(l, -1) for l in labels])

def train_test_split(X, y, test_ratio=0.2):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_at = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split_at], indices[split_at:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

if __name__ == "__main__":
    # Dynamically find project root and data paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    class_path = os.path.join(base_dir, 'datasets', 'classification_data.txt')

    texts, labels = load_classification_data(class_path)
    vocab = build_vocab(texts)
    seqs = texts_to_sequences(texts, vocab)
    X = pad_sequences(seqs, maxlen=20)
    y = encode_labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
def load_generation_corpus(path):
    """Read the full text corpus as a single lowercase string."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().lower()

def build_gen_sequences(text, vocab, seq_len=10):
    """
    From a tokenized text (string), produce:
      X: list of sequences (each a list of word‑indices of length seq_len)
      y: list of target indices (the word after each sequence)
    """
    words = text.split()
    X, y = [], []
    for i in range(len(words) - seq_len):
        seq = words[i : i + seq_len]
        target = words[i + seq_len]
        X.append([vocab.get(w, vocab['<UNK>']) for w in seq])
        y.append(vocab.get(target, vocab['<UNK>']))
    return np.array(X), np.array(y)
