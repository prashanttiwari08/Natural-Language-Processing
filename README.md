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
