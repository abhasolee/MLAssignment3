# Assignment 3: Natural Language Processing

## 1. Project Overview
This project covers two core Natural Language Processing (NLP) tasks: Text Generation or Sequence Completion and Machine Translation. We evaluate and compare two different RNNs, Long Short-Term Memory (LSTM) vs. Gated Recurrent Unit (GRU) and two different type of word/token embeddings, Word2Vec vs. One-Hot Encoding. 

## 2. Dataset Description
* **Task 1 (Text Generation):** We utilized the **Shakespeare dataset**, which is a simple dataset for sequence generation. The data was processed at the word level to predict the next word in a sequence.
* **Task 2 (Machine Translation):** We utilized the **Multi30k dataset** to train the English-to-German translations. Data pre-processing was done by lowercasing text, bounding seqeunces with `<SOS>` and `<EOS>` tokens, and applying `<PAD>` tokens to unify batch sequence lengths to a maxium of 20 tokens.

## 3. Model Architectures Used
We used to different Recurrent Neural Networks (RNNs):
* **LSTM (Long Short-Term Memory):** Used for text generation and as the foundational architecture for the Encoder-Decoder translation model. LSTMs utilize input, forget, and output gates to manage long-term dependencies.
* **GRU (Gated Recurrent Unit):** Implemented as a comparative alternative. GRUs merge the forget and input gates into a single "update gate", making the architecture computationally lighter and faster to train.

## 4. Word Embedding Methods
Two fundamentally different methods were used for this process:
* **One-Hot Encoding:** A baseline approach where each token is represented as a sparse, high-dimensional vector consisting of all zeros and a single `1` corresponding to its vocabulary index.
* **Word2Vec Embeddings:** Dense, continuous vector representations trained using Gensim. These embeddings capture semantic relationships between words, placing contextually similar words closer together in the vector space.

## 5. Experimental Results
Models were trained for 3-5 epochs (Task1 for 3 epochs and Task2 for 5 epochs) on subsampled datasets to evaluate results

### Task 1: Text Generation
| Model Setup | Evaluation Metric (Perplexity) |
| :--- | :--- |
| **LSTM + Word2Vec** | 24.2928 |
| **LSTM + One-Hot** | 21.1833 |
| **GRU + Word2Vec** | 21.7199 |
| **GRU + One-Hot** | **10.5334** |

### Task 2: Machine Translation
| Model Setup | Evaluation Metric (BLEU Score) | Final Loss |
| :--- | :--- | :--- |
| **Seq2Seq LSTM + Word2Vec** | 0.0456 | 3.5794 |
| **Seq2Seq LSTM + One-Hot** | 0.0455 | 3.6559 |
| **Seq2Seq GRU + Word2Vec** | **0.1121** | **3.0160** |
| **Seq2Seq GRU + One-Hot** | 0.0432 | 3.5191 |

## 6. Comparison of Models
From our results, GRU with One Hot Encoding is the best for text generation and GRU with Word2Vec embedding is the best for machine translation.
* **GRU vs. LSTM:** Across both tasks, the GRU models consistently outperformed the LSTMs. In Task 2, the GRU with Word2Vec embedding achieved a BLEU score of 0.1121, more than double the LSTM models. Because GRUs possess fewer parameters and a simpler gating mechanism, they typically converge faster than LSTMs on smaller vocabulary sizes and shorter training durations. 
* **One-Hot vs. Word2Vec:** The advantages of dense semantic embeddings is apparent in the Machine Translation task. The GRU with Word2Vec embedding dramatically outperformed the GRU with One-Hot encoding (0.1121 vs 0.0432 BLEU score). Because Word2Vec groups semantically similar words together in continuous vector space, the neural network experiences smoother gradient updates when learning the mapping between English and German semantics. In contrast, predicting targets from the orthogonal, equidistant axes of sparse One-Hot vectors become an obstacle fot the model to learn generalized translation rules.

## 7. Challenges Faced During Implementation
* **Padding and Loss Calculation:** In Task 2, padding sentences to uniform lengths introduced artificial `<PAD>` tokens. These tokens artificially inflated the model's accuracy since it is easier to correctly predict a `<PAD>` token. To fix this, the `CrossEntropyLoss` function was strictly configured to ignore the padding index (index = 0) to skip `<PAD>` tokens entirely during loss calculation.
* **Out of Vocabulary Tokens:** Both One-Hot and Word2Vec models struggled with words that appeared in the testing subsets but not in our capped training vocabularies. Creating a dedicated `<UNK>` (unknown) token and randomizing its embedding vector was necessary to prevent runtime crashes and index-out-of-bounds errors.

## 8. Limitations of the Considered Models
* **Sequential Bottleneck:** Both LSTMs and GRUs process tokens strictly one after the other. This sequential nature prevents parallelization during training, making them incredibly slow to scale on large corpora like the full Multi30k compared to modern architectures.
* **Context Vector Decay:** In our standard Seq2Seq model, the entire English source sentence is compressed into a single, fixed-size hidden context vector. For longer sentences, information from the beginning of the English sentence inevitably decays and is lost by the time the Decoder starts generating the German output.

## 9. Possible Future Improvements
* **Implementing Attention Mechanisms:** The most necessary improvement for the translation task is adding Bahdanau or Luong Attention mechanisms. This would allow the Decoder to dynamically "look back" at specific, relevant words in the Encoder's hidden states (resolving the context decay limitation) rather than relying purely on the final state vector.
* **Migration to Transformers:** Replacing the RNN backbones entirely with Transformer architectures (utilizing multi-head self-attention) would eliminate the sequential processing bottleneck and drastically improve the models' understanding of long-range bilingual grammatical syntax.

## 10. How to run the code

1) Clone the Github repo.  
2) Install the required dependencies present in the requirements.txt file

```py
pip install -r requirements.txt
```

3) Run the train_task1 and train_task2 python files.

```py
python -u train_task1.py
python -u train_task2.py
```