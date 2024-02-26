# sentence-swissbert

<!-- Provide a quick summary of what the model is/does. -->

The [SwissBERT](https://huggingface.co/ZurichNLP/swissbert) model was finetuned via self-supervised [SimCSE](http://dx.doi.org/10.18653/v1/2021.emnlp-main.552) (Gao et al., EMNLP 2021) for sentence embeddings, using ~1.5 million Swiss news articles from up to 2023 retrieved via [Swissdox@LiRI](https://t.uzh.ch/1hI). Following the [Sentence Transformers](https://huggingface.co/sentence-transformers) approach (Reimers and Gurevych,
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
model_name = "jgrosjean-mathesis/sentence-swissbert"
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

    # Extract sentence embeddings via mean pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask

    return embedding

# Try it out
sentence_0 = "Wir feiern am 1. August den Schweizer Nationalfeiertag."
sentence_0_embedding = generate_sentence_embedding(sentence_0, language="de")
print(sentence_0_embedding)
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
The sentence swissBERT model has been trained on news articles only. Hence, it might not perform as well on other text classes. Furthermore, it is specific to a Switzerland-related context, which means it probably does not perform as well on text that does not fall in that category. Additionally, the model has neither been trained nor evaluated for machine translation tasks.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

German, French, Italian and Romansh documents in the [Swissdox@LiRI database](https://t.uzh.ch/1hI) up to 2023.

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

This model was finetuned via self-supervised [SimCSE](http://dx.doi.org/10.18653/v1/2021.emnlp-main.552). The positive sequence pairs consist of the article body vs. its title and lead, wihout any hard negatives.

The fine-tuning script can be accessed [here](https://github.com/jgrosjean-mathesis/sentence-swissbert/tree/main/training).

#### Training Hyperparameters

- Number of epochs: 1
- Learning rate: 1e-5
- Batch size: 512
- Temperature: 0.05

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The two evaluation tasks make use of the [20 Minuten dataset](https://www.zora.uzh.ch/id/eprint/234387/) compiled by Kew et al. (2023), which contains Swiss news articles with topic tags and summaries. Parts of the dataset were automatically translated to French, Italian using a Google Cloud API and to Romash via a [Textshuttle](https://textshuttle.com/en) API.

#### Evaluation via Semantic Textual Similarity

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Embeddings are computed for the summary and content of each document. Subsequently, the embeddings are matched by maximizing cosine similarity scores between each summary and content embedding pair.

The performance is measured via accuracy, i.e. the ratio of correct vs. total matches. The script can be found [here](https://github.com/jgrosjean-mathesis/sentence-swissbert/tree/main/evaluation).


#### Evaluation via Text Classification

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Articles with the topic tags "movies/tv series", "corona" and "football" (or related) are filtered from the corpus and split into training data (80%) and test data (20%). Subsequently, embeddings are set up for the train and test data. The test data is then classified using the training data via a k-nearest neighbors approach. The script can be found [here](https://github.com/jgrosjean-mathesis/sentence-swissbert/tree/main/evaluation).

Note: For French, Italian and Romansh, the training data remains in German, while the test data comprises of translations. This provides insights in the model's abilities in cross-lingual transfer.

### Results

Sentence SwissBERT achieves comparable or better results as the best-performing multilingual Sentence-BERT model in these tasks (distiluse-base-multilingual-cased). It outperforms it in all evaluation task, except for the text classification in Italian.

| Evaluation task        |Swissbert |           |Sentence Swissbert|           |Sentence-BERT|           |
|------------------------|----------|-----------|------------------|-----------|-------------|-----------|
|                        |accuracy  |f1-score   |accuracy          |f1-score   |accuracy     |f1-score   |
| Semantic Similarity DE | 87.20 %  | -         |**93.40 %**       |    -      |  91.80 %    |    -      |
| Semantic Similarity FR | 84.97 %  | -         |**93.99 %**       |    -      |  93.19 %    |    -      |
| Semantic Similarity IT | 84.17 %  | -         |**92.18 %**       |    -      |  91.58 %    |    -      |
| Semantic Similarity RM | 83.17 %  | -         |**91.58 %**       |    -      |  73.35 %    |    -      |
| Text Classification DE |          |        %  |                  |**78.49 %**|             |  77.23 %  |
| Text Classification FR |          |        %  |                  |**77.18 %**|             |  76.83 %  |
| Text Classification IT |          |        %  |                  |  76.65 %  |             |**76.90 %**|
| Text Classification RM |          |        %  |                  |**77.20 %**|             |  65.35 %  |

#### Baseline

The baseline uses mean pooling embeddings from the last hidden state of the original swissbert model and (in this task) best-performing Sentence-BERT model [distiluse-base-multilingual-cased-v1](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)
