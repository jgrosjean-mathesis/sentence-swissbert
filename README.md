# swissbert-for-sentence-embeddings

<!-- Provide a quick summary of what the model is/does. -->

The [SwissBERT](https://huggingface.co/ZurichNLP/swissbert) model was finetuned via unsupervised [SimCSE](http://dx.doi.org/10.18653/v1/2021.emnlp-main.552) (Gao et al., EMNLP 2021) for sentence embeddings, using ~1 million Swiss news articles published in 2022 from [Swissdox@LiRI](https://t.uzh.ch/1hI). Following the [Sentence Transformers](https://huggingface.co/sentence-transformers) approach (Reimers and Gurevych,
2019), the average of the last hidden states (pooler_type=avg) is used as sentence representation.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6564ab8d113e2baa55830af0/zUUu7WLJdkM2hrIE5ev8L.png)

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** [Juri Grosjean](https://huggingface.co/jgrosjean)
- **Model type:** [XMOD](https://huggingface.co/facebook/xmod-base)
- **Language(s) (NLP):** de_CH, fr_CH, it_CH, rm_CH
- **License:** Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- **Finetuned from model:** [SwissBERT](https://huggingface.co/ZurichNLP/swissbert)

## Use

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

```python
import torch

from transformers import AutoModel, AutoTokenizer

# Load swissBERT for sentence embeddings model
model_name = "jgrosjean-mathesis/swissbert-for-sentence-embeddings"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_sentence_embedding(sentence, language):

    # Set adapter to specified language
    if "de" in language:
        model.set_default_language("de_CH")
    if "fr" in language:
        model.set_default_language("fr_CH")
    if "it" in language:
        model.set_default_language("it_CH")
    if "rm" in language:
        model.set_default_language("rm_CH")
    
    # Tokenize input sentence
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Take tokenized input and pass it through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract average sentence embeddings from the last hidden layer
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding

sentence_embedding = generate_sentence_embedding("Wir feiern am 1. August den Schweizer Nationalfeiertag.", language="de")
print(sentence_embedding)
```
Output:
```
tensor([[ 5.6306e-02, -2.8375e-01, -4.1495e-02,  7.4393e-02, -3.1552e-01,
          1.5213e-01, -1.0258e-01,  2.2790e-01, -3.5968e-02,  3.1769e-01,
          1.9354e-01,  1.9748e-02, -1.5236e-01, -2.2657e-01,  1.3345e-02,
        ...]])
```

### Semantic Textual Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Define two sentences
sentence_1 = ["Der Zug kommt um 9 Uhr in Zürich an."]
sentence_2 = ["Le train arrive à Lausanne à 9h."]

# Compute embedding for both
embedding_1 = generate_sentence_embedding(sentence_1, language="de")
embedding_2 = generate_sentence_embedding(sentence_2, language="fr")

# Compute cosine-similarity
cosine_score = cosine_similarity(embedding_1, embedding_2)

# Output the score
print("The cosine score for", sentence_1, "and", sentence_2, "is", cosine_score)
```
Output:
```
The cosine score for ['Der Zug kommt um 9 Uhr in Zürich an.'] and ['Le train arrive à Lausanne à 9h.'] is [[0.85555995]]
```

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
This model has been trained on news articles only. Hence, it might not perform as well on other text classes.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

German, French, Italian and Romansh documents in the [Swissdox@LiRI database](https://t.uzh.ch/1hI) from 2022.

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

This model was finetuned via unsupervised [SimCSE](http://dx.doi.org/10.18653/v1/2021.emnlp-main.552). The same sequence is passed to the encoder twice and the distance between the two resulting embeddings is minimized.  Because of the drop-out, it will be encoded at slightly different positions in the vector space.

The fine-tuning script can be accessed [here](Link).

#### Training Hyperparameters

- Number of epochs: 1
- Learning rate: 1e-5
- Batch size: 512

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

#### Baseline

The first baseline is [distiluse-base-multilingual-cased](https://www.sbert.net/examples/training/multilingual/README.html), a high-performing Sentence Transformer model that is able to process German, French and Italian (and more).

The second baseline uses mean pooling embeddings from the last hidden state of the original swissbert model.

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The two evaluation tasks make use of the [20 Minuten dataset](https://www.zora.uzh.ch/id/eprint/234387/) compiled by Kew et al. (2023), which contains Swiss news articles with topic tags and summaries. Parts of the dataset were automatically translated to French and Italian using a Google Cloud API.

#### Evaluation via Semantic Textual Similarity

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Embeddings are computed for the summary and content of each document. Subsequently, the embeddings are matched by minimizing cosine similarity scores betweend each summary and content embedding pair.

The performance is measured via accuracy, i.e. the ratio of correct vs. incorrect matches.


#### Evaluation via Text Classification

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Articles with the topic tags "movies/tv series", "corona" and "football" (or related) are filtered from the corpus and split into training data (80%) and test data (20%). Subsequently, embeddings are set up for the train and test data. The test data is then classified using the training data via a k-nearest neighbor approach.

Note: For French and Italian, the training data remains in German, while the test data comprises of translations. This provides insights in the model's abilities in cross-lingual transfer.

### Results

Making use of an unsupervised training approach, Swissbert for Sentence Embeddings achieves comparable results as the best-performing multilingual Sentence-BERT model in the semantic textual similarity task for German and outperforms it in the French text classification task.

| Evaluation task        |swissbert |         |swissbert for SE|         |Sentence-BERT|         |
|------------------------|----------|---------|----------------|---------|-------------|---------|
|                        |accuracy  |f1-score |accuracy        |f1-score |accuracy     |f1-score |
| Semantic Similarity DE | 83.80    | -       |**87.70**       |    -    |**87.70**    | -       |
| Semantic Similarity FR | 82.30    | -       | 84.02          |    -    |**91.10**    | -       |
| Semantic Similarity IT | 83.00    | - `     | 84.00          |    -    |**89.80**    | -       |
| Text Classification DE | 95.76    |**91.99**| 94.70          |  89.43  |  95.61      | 91.20   |
| Text Classification FR | 94.55    | 88.52   | 95.30          |**89.91**|  94.55      | 89.82   |
| Text Classification IT | 93.48    | 88.29   | 94.85          |  90.36  |  95.91      |**92.05**|
