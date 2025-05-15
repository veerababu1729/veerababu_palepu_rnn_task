import os, sys

# Ensure project root is on Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# 1. Import our preprocessing helpers
from utils.preprocessing import (
    load_classification_data,
    build_vocab,
    texts_to_sequences,
    pad_sequences,
    encode_labels,
    train_test_split
)

def main():
    # --- Paths ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'datasets', 'classification_data.txt')

    # --- Load & preprocess ---
    texts, labels = load_classification_data(data_path)
    vocab = build_vocab(texts)
    sequences = texts_to_sequences(texts, vocab)
    maxlen = 20
    X = pad_sequences(sequences, maxlen)
    y = encode_labels(labels)
    y = to_categorical(y, num_classes=3)      # one‑hot encode labels

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

    # --- Build model ---
    vocab_size = len(vocab)
    embed_dim = 50
    rnn_units = 32

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen),
        SimpleRNN(rnn_units),
        Dense(3, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    # --- Train ---
    epochs = 10
    batch_size = 8
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

    # --- Evaluate ---
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Optional: save model
    model.save(os.path.join(base_dir, 'src', 'classification_rnn.h5'))

if __name__ == "__main__":
    main()
