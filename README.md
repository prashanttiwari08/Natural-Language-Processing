# NLP Lab Work

Hands-on Natural Language Processing labs in Python, going from raw text cleaning to transformer-based sentiment analysis.

## Labs

### Lab 1 – Text Preprocessing (SMS Spam Dataset)
- Loaded and cleaned the SMS Spam dataset with **pandas**
- Built a text-cleaning pipeline: lowercasing, contraction expansion, URL/number/punctuation removal
- Applied **tokenization**, **stopword removal**, **stemming** (Porter) and **lemmatization** (spaCy)
- Libraries: `pandas`, `nltk`, `spaCy`, `contractions`

### Lab 2 – Word Embeddings (Shakespeare's *Hamlet*)
- Preprocessed text from the NLTK Gutenberg corpus
- Compared three embedding approaches:
  - **Word2Vec** – trained on the text (Skip-gram) using `gensim`
  - **GloVe** – pretrained `glove-wiki-gigaword-100` vectors
  - **BERT** – contextual embeddings from `bert-base-uncased` (Hugging Face)
- Libraries: `gensim`, `transformers`, `torch`, `nltk`

### Lab 3 – BERT Sentiment Analysis & ABSA
- Sentiment classification with pretrained **BERT** (`bert-base-uncased-SST-2`) via Hugging Face pipelines
- Visualized prediction confidence and tested hard cases: negation, sarcasm, mixed sentiment
- Explored **Aspect-Based Sentiment Analysis (ABSA)** for mixed reviews like *"I love chicken but hate KFC"*
- Simple automatic aspect extraction and a final dashboard
- Libraries: `transformers`, `torch`, `spaCy`, `matplotlib`, `seaborn`

### Lab 4 – Implementing a GPT-2 Model from Scratch (PyTorch)
- Built the **GPT-2 (124M) architecture** from the ground up in **PyTorch**, one component at a time
- Used the GPT-2 **BPE tokenizer** (Hugging Face) to convert text into token IDs
- Implemented each building block and tested its output shapes:
  - **Token Embedding** and **Positional Encoding** (sine/cosine)
  - **Multi-Head Attention** (12 heads)
  - **Layer Normalization**
  - **Feed-Forward Network** (GELU activation)
  - **Transformer Block**, combining attention, layer norm, feed-forward, and dropout
  - **Output Layer** that produces logits over the 50,257-token vocabulary
- Assembled the blocks into a full model of 12 layers, 768-dim embeddings, and a 1024-token context
- Wrote a simple text-generation loop that predicts the next 5 tokens from a prompt ("Every day is your"), using the untrained model
- Libraries: `torch`, `transformers`, `math`
