import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import load_model

# 1. Ensure we can import from utils
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from utils.preprocessing import load_generation_corpus, build_vocab

def main():
    # 2. Parse a seed sentence from the command line
    parser = argparse.ArgumentParser(
        description="Next‐Word Generation Test (seed must be ≥10 words)"
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="Your starting sentence (at least 10 words). If omitted, uses the first 10 words of the corpus.",
        default=None
    )
    args = parser.parse_args()

    # 3. Load corpus & rebuild vocab
    data_path = os.path.join(base_dir, 'datasets', 'generation_data.txt')
    text = load_generation_corpus(data_path)
    vocab = build_vocab([text])
    inv_vocab = {i: w for w, i in vocab.items()}

    seq_len = 10   # must match your training sequence length

    # 4. Determine seed words
    if args.seed:
        words = args.seed.lower().split()
        if len(words) < seq_len:
            print(f"Error: Seed must be at least {seq_len} words. You provided {len(words)}.")
            return
        seed_words = words[:seq_len]
    else:
        # default: take first seq_len words from corpus
        seed_words = text.split()[:seq_len]

    seed_seq = np.array([[vocab.get(w, vocab['<UNK>']) for w in seed_words]])
    generated = seed_words.copy()

    # 5. Load your trained model
    model = load_model(os.path.join(base_dir, 'src', 'generation_rnn.h5'))

    # 6. Generate 20 new words
    for _ in range(20):
        preds = model.predict(seed_seq, verbose=0)[0]
        idx = int(np.argmax(preds))
        next_word = inv_vocab.get(idx, '<UNK>')
        generated.append(next_word)
        # slide window
        seed_seq = np.roll(seed_seq, -1, axis=1)
        seed_seq[0, -1] = idx

    # 7. Print results
    print("\nSeed Sentence:")
    print(" ".join(seed_words))
    print("\nGenerated Next 20 Words:")
    print(" ".join(generated[seq_len:]))

    # 8. Brief coherence comment
    print("\n[Coherence Check]")
    print("→ Does the continuation follow logically? You should comment here in your submission.")

if __name__ == "__main__":
    main()
