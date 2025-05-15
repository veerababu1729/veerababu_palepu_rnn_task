```markdown
# Simple RNN NLP Assignment

**Author:** Veerababu Palepu  
**Branch:** `veerababu_palepu_rnn_task`  
**Date:** 2025‑05‑14  

---

## 📁 Project Structure

```

veerababu\_palepu\_rnn\_task/
├── datasets/
│   ├── classification\_data.txt    # Labeled snippets (Math/Science/History)
│   └── generation\_data.txt        # Multi‑paragraph corpus for next‑word generation
├── src/
│   ├── classification\_model.py    # Build, train & evaluate SimpleRNN classifier
│   └── generation\_model.py        # Build, train & generate with SimpleRNN
├── utils/
│   └── preprocessing.py           # Load/tokenize/sequence/pad/split helpers
├── test\_classification.py         # Classifier inference on custom snippets
├── test\_generation.py             # Generator inference on a 10‑word seed → 20 words
├── .gitignore                     # Excludes virtual‑env, caches, large files
└── README.md                      # This file

````

---

## ⚙️ Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/veerababu1729/veerababu_palepu_rnn_task.git
   cd veerababu_palepu_rnn_task
````

2. **Create & activate** a Python 3 virtual environment

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --no-cache-dir numpy matplotlib tensorflow==2.13.0
   ```

---

## 🚀 How to Run

All commands assume your current directory is the project root.

### 1. Train & Evaluate the Classifier

```bash
python src/classification_model.py
```

* **Output:**

  * Prints model summary
  * Training/validation loss & accuracy per epoch
  * Final test accuracy (on a held‑out 20% split)
  * Saved model checkpoint: `src/classification_rnn.h5`

### 2. Train & Demonstrate Next‑Word Generation

```bash
python src/generation_model.py
```

* **Output:**

  * Prints model summary
  * Training loss per epoch
  * “Generated Text:” seed + 20 new words
  * Saved model checkpoint: `src/generation_rnn.h5`

---

## 🔍 Inference & Testing

#### A) Classification Inference

```bash
python test_classification.py
```

* **What it does:**

  * Loads `classification_rnn.h5`
  * Preprocesses hardcoded sample snippets
  * Prints predicted label + confidence for each

#### B) Generation Inference

```bash
# Default: uses the first 10 words of generation_data.txt
python test_generation.py

# Or supply a custom ≥10‑word seed:
python test_generation.py \
  --seed "For example solving the equation 3x 5 20 requires isolating x by subtracting"
```

* **What it does:**

  * Loads `generation_rnn.h5`
  * Preprocesses your seed (10 words)
  * Generates and prints 20 subsequent words
  * Prompts you to add a brief coherence comment

---

## 🔧 Technical Details

* **Environment:**

  * Python 3.10+
  * TensorFlow 2.13.0 (`tensorflow.keras`)
  * NumPy 1.24+
  * Matplotlib 3.7+ (used only for any optional plots)

* **Preprocessing:**

  * **Classification:**

    * Tokenize on whitespace, lowercase all text
    * Vocabulary: `{<PAD>:0, <UNK>:1, word_i:2…}`
    * Sequences padded/truncated to length 20
    * Labels mapped: Math→0, Science→1, History→2 (one‑hot encoded)
    * Train/test split: 80/20
  * **Generation:**

    * Full corpus lowercased & tokenized
    * Sliding window of `seq_len=10` words → next‑word target
    * Inputs: shape `(num_examples, 10)`
    * Targets: one‑hot vectors of length `vocab_size`

* **Model Architectures:**

  * **Embedding Layer:**

    * Input dim = vocabulary size
    * Output dim = 50
  * **SimpleRNN Layer:**

    * Classification: 32 units
    * Generation: 64 units
  * **Dense Output:**

    * Classification: 3 units + softmax
    * Generation: `vocab_size` units + softmax

* **Training:**

  * Optimizer: Adam (default parameters)
  * Loss: categorical\_crossentropy
  * Metrics: accuracy (classification only)
  * Epochs: 10 for classification, 20 for generation
  * Batch sizes: 8 (classification), 16 (generation)

* **Model Persistence:**

  * Saved in HDF5 (`.h5`) format
  * Reload via `keras.models.load_model`

---

## 📌 Notes

* Adjust hyperparameters in `src/*.py` as needed.
* Always run scripts from the project root to resolve relative paths.
* To reproduce or share the environment, generate:

  ```bash
  pip freeze > requirements.txt
  ```

---


## 🧠 How the Models Work

### 🔹 1. Educational Text Classification (RNN Classifier)

**Goal:**
Predict whether a given educational sentence belongs to the *Math*, *Science*, or *History* category.

**Pipeline:**

1. **Text Input:**
   Example:
   `"Photosynthesis converts sunlight into chemical energy in plants."`

2. **Tokenization & Vocabulary Building:**

   * The text is split into lowercase words.
   * A vocabulary dictionary assigns each unique word an index.
   * Unknown words → `<UNK>`, padded sequences → `<PAD>`

3. **Sequence Encoding:**

   * Text is converted into a fixed-length sequence of integers (length = 20).
   * Example:
     `"photosynthesis converts sunlight..."` → `[45, 21, 90, ..., 0, 0]` (padded)

4. **Model Architecture:**

   * **Embedding Layer**
     Converts each word index into a 50‑dimensional dense vector.
   * **SimpleRNN Layer (32 units)**
     Processes the sequence step by step, learning temporal relationships.
   * **Dense Layer (softmax)**
     Outputs probabilities over 3 categories.

5. **Training:**

   * Uses categorical cross-entropy loss + Adam optimizer.
   * Evaluates accuracy on a held-out 20% test set.

6. **Output:**

   * The model predicts a probability distribution over the labels.
   * Highest probability label is selected.

   Example Output:

   ```
   "Photosynthesis converts..." → Predicted: Science (conf: 0.92)
   ```

---

### 🔹 2. Next Word Generation (RNN Generator)

**Goal:**
Given a starting sentence of 10 words, generate the next 20 words to simulate content auto-completion.

**Pipeline:**

1. **Corpus Input:**
   Example content from `generation_data.txt`:

   ```
   Algebra is the branch of mathematics dealing with symbols...
   ```

2. **Tokenization & Vocabulary Building:**

   * The full paragraph is tokenized into words and indexed.
   * A sliding window of 10 words is used to predict the 11th word.

   Example:

   ```
   Input:  ["algebra", "is", "the", ..., "symbols"]  
   Target: "and"
   ```

3. **Model Architecture:**

   * **Embedding Layer**
     Maps word indices to dense vectors.
   * **SimpleRNN Layer (64 units)**
     Processes the 10‑word sequence.
   * **Dense Layer (softmax)**
     Outputs probabilities over the vocabulary → predicts the next word.

4. **Training:**

   * Trained for 20 epochs on many 10→1 word pairs from the corpus.

5. **Inference (Generation):**

   * Start with a seed sentence (10 words).
   * Predict 1 word → append it → slide window → predict next → repeat 20 times.

6. **Output:**

   * A full 30-word sequence: original 10 + predicted 20.

   Example Output:

   ```
   Seed:      For example solving the equation 3x 5 20 requires isolating x by subtracting  
   Predicted: 5 and dividing by 3 advanced topics include polynomials factorization and...
   ```

---

### 🔬 Why RNN?

* Recurrent Neural Networks are good at handling **sequential data** like text.
* They process words in order, preserving **context** using hidden states.
* This makes them suitable for both **classification of text sequences** and **generation of new sequences**.

---
Certainly! Here's a precise explanation you can **add to your README** under a new section titled:

---

## 🧠 How the Models Work

### 🔹 1. Educational Text Classification (RNN Classifier)

**Goal:**
Predict whether a given educational sentence belongs to the *Math*, *Science*, or *History* category.

**Pipeline:**

1. **Text Input:**
   Example:
   `"Photosynthesis converts sunlight into chemical energy in plants."`

2. **Tokenization & Vocabulary Building:**

   * The text is split into lowercase words.
   * A vocabulary dictionary assigns each unique word an index.
   * Unknown words → `<UNK>`, padded sequences → `<PAD>`

3. **Sequence Encoding:**

   * Text is converted into a fixed-length sequence of integers (length = 20).
   * Example:
     `"photosynthesis converts sunlight..."` → `[45, 21, 90, ..., 0, 0]` (padded)

4. **Model Architecture:**

   * **Embedding Layer**
     Converts each word index into a 50‑dimensional dense vector.
   * **SimpleRNN Layer (32 units)**
     Processes the sequence step by step, learning temporal relationships.
   * **Dense Layer (softmax)**
     Outputs probabilities over 3 categories.

5. **Training:**

   * Uses categorical cross-entropy loss + Adam optimizer.
   * Evaluates accuracy on a held-out 20% test set.

6. **Output:**

   * The model predicts a probability distribution over the labels.
   * Highest probability label is selected.

   Example Output:

   ```
   "Photosynthesis converts..." → Predicted: Science (conf: 0.92)
   ```

---

### 🔹 2. Next Word Generation (RNN Generator)

**Goal:**
Given a starting sentence of 10 words, generate the next 20 words to simulate content auto-completion.

**Pipeline:**

1. **Corpus Input:**
   Example content from `generation_data.txt`:

   ```
   Algebra is the branch of mathematics dealing with symbols...
   ```

2. **Tokenization & Vocabulary Building:**

   * The full paragraph is tokenized into words and indexed.
   * A sliding window of 10 words is used to predict the 11th word.

   Example:

   ```
   Input:  ["algebra", "is", "the", ..., "symbols"]  
   Target: "and"
   ```

3. **Model Architecture:**

   * **Embedding Layer**
     Maps word indices to dense vectors.
   * **SimpleRNN Layer (64 units)**
     Processes the 10‑word sequence.
   * **Dense Layer (softmax)**
     Outputs probabilities over the vocabulary → predicts the next word.

4. **Training:**

   * Trained for 20 epochs on many 10→1 word pairs from the corpus.

5. **Inference (Generation):**

   * Start with a seed sentence (10 words).
   * Predict 1 word → append it → slide window → predict next → repeat 20 times.

6. **Output:**

   * A full 30-word sequence: original 10 + predicted 20.

   Example Output:

   ```
   Seed:      For example solving the equation 3x 5 20 requires isolating x by subtracting  
   Predicted: 5 and dividing by 3 advanced topics include polynomials factorization and...
   ```

---

### 🔬 Why RNN?

* Recurrent Neural Networks are good at handling **sequential data** like text.
* They process words in order, preserving **context** using hidden states.
* This makes them suitable for both **classification of text sequences** and **generation of new sequences**.

---
Thank you for reviewing my implementation!
Feel free to explore further by swapping in LSTM/GRU layers or expanding the dataset.

