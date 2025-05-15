import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# Ensure project root on path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from utils.preprocessing import (
    load_generation_corpus,
    build_vocab,
    build_gen_sequences
)

def main():
    # --- Paths & params ---
    data_path = os.path.join(base_dir, 'datasets', 'generation_data.txt')
    seq_len = 10      # number of input words
    embed_dim = 50
    rnn_units = 64
    epochs = 20
    batch_size = 16

    # --- Load & preprocess ---
    text = load_generation_corpus(data_path)
    vocab = build_vocab([text])            # reuse same vocab builder
    X, y = build_gen_sequences(text, vocab, seq_len)
    vocab_size = len(vocab)
    y = to_categorical(y, num_classes=vocab_size)

    # --- Build model ---
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_len),
        SimpleRNN(rnn_units),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    # --- Train ---
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # --- Demonstrate generation ---
    # Pick a seed sequence (first seq_len words of the corpus)
    seed_words = text.split()[:seq_len]
    seed_seq = np.array([[vocab.get(w, vocab['<UNK>']) for w in seed_words]])
    generated = seed_words.copy()

    for _ in range(20):  # generate 20 words
        preds = model.predict(seed_seq, verbose=0)[0]
        next_index = np.argmax(preds)
        # map index back to word
        next_word = [w for w,i in vocab.items() if i == next_index]
        next_word = next_word[0] if next_word else '<UNK>'
        generated.append(next_word)
        # shift window
        seed_seq = np.roll(seed_seq, -1, axis=1)
        seed_seq[0, -1] = next_index

    print("\nGenerated Text:")
    print(" ".join(generated))

    # Optional: save model
    model.save(os.path.join(base_dir, 'src', 'generation_rnn.h5'))

if __name__ == "__main__":
    main()
