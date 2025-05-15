import os, sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# add project root so utils can be found
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from utils.preprocessing import (
    load_classification_data,
    build_vocab,
    texts_to_sequences,
    pad_sequences,
    encode_labels
)

# 1. Load vocab built from your training data
texts, labels = load_classification_data(os.path.join(base_dir, 'datasets', 'classification_data.txt'))
vocab = build_vocab(texts)
maxlen = 20

# 2. Load the trained model
model = load_model(os.path.join(base_dir, 'src', 'classification_rnn.h5'))

# 3. Prepare your own sample(s)
samples = [
    "The mitochondria is the powerhouse of the cell",
    "Solve for y in the equation y - 4 = 10",
    "Napoleon was defeated at Waterloo in 1815",
    "newton is a"
]

# 4. Preprocess
seqs = texts_to_sequences([s.lower() for s in samples], vocab)
X = pad_sequences(seqs, maxlen)

# 5. Predict & map back to labels
preds = model.predict(X)
label_map = {0: 'Math', 1: 'Science', 2: 'History'}
for text, p in zip(samples, preds):
    chosen = np.argmax(p)
    print(f"\"{text}\"  →  {label_map[chosen]}  (conf={p[chosen]:.2f})")
