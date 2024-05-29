import os
import torch
import argparse
import logging
import pandas as pd
import plotly.io as pio
import numpy as np
import gensim.corpora as corpora
from transformers import AutoModel, AutoTokenizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel

def parse_args():
    """parses command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="specify the name of the model via --model (sentence-bert, swissbert, sentence-swissbert)")
    parser.add_argument("--input_file", type=str, required=True, help="specify the name or path of the input file via --input_file")
    parser.add_argument("--language", type=str, help="specify language as de_ch, fr_ch, it_ch, or rm_ch")
    return parser.parse_args()

def set_up_language_adapter(swissbert_model, language):
    if "de" in language:
        swissbert_model.set_default_language("de_CH")
        print("Language set to de_CH")
    elif "fr" in language:
        swissbert_model.set_default_language("fr_CH")
        print("Language set to fr_CH")
    elif "it" in language:
        swissbert_model.set_default_language("it_CH")
        print("Language set to it_CH")
    elif "rm" in language:
        swissbert_model.set_default_language("rm_CH")
        print("Language set to rm_CH")
    else:
        print("Invalid language. Please specify a valid language via --language (de_ch, fr_ch, it_ch, rm_ch)")
        exit()

def generate_swissbert_embedding(sentence, sentence_model, sentence_tokenizer):
    """generates sentence embeddings using SwissBERT"""
    inputs = sentence_tokenizer(sentence, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = sentence_model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    embedding = embedding.numpy()
    return embedding

def set_up_model(model_name, language):
    """sets up the sentence and topic model"""
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    if "swissbert" in model_name:
        topic_model = BERTopic(ctfidf_model=ctfidf_model, nr_topics=21, top_n_words =15, calculate_probabilities=True)
        if model_name == "swissbert":
            sentence_model = AutoModel.from_pretrained("zurichNLP/swissbert")
            sentence_tokenizer = AutoTokenizer.from_pretrained("zurichNLP/swissbert")
            set_up_language_adapter(sentence_model, language)
            return topic_model, sentence_model, sentence_tokenizer
        elif model_name == "sentence-swissbert":
            sentence_model = AutoModel.from_pretrained("jgrosjean-mathesis/sentence-swissbert")
            sentence_tokenizer = AutoTokenizer.from_pretrained("jgrosjean-mathesis/sentence-swissbert")
            set_up_language_adapter(sentence_model, language)
            return topic_model, sentence_model, sentence_tokenizer
    elif model_name == "sentence-bert":
        sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        topic_model = BERTopic(embedding_model=sentence_model, ctfidf_model=ctfidf_model, nr_topics=22, top_n_words =15, calculate_probabilities=True)
        return topic_model
    else:
        print("Invalid model name. Please specify a valid model name (sentence-bert, swissbert, sentence-swissbert)")
        exit()
   
def process_input_file(input_file):
    """Processes input file"""
    with open(input_file, "r") as file:
        documents = file.readlines()
    return documents

def calculate_coherence(topic_model, topics, documents):
    """"calculates topic coherence scores for assessment, copied from https://github.com/MaartenGr/BERTopic/issues/90"""
    documents = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    tokens = [analyzer(doc) for doc in cleaned_docs]

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    topic_words = [[words for words, _ in topic_model.get_topic(topic)] for topic in range(len(set(topics))-1)]

    umass_coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    umass_coherence = umass_coherence_model.get_coherence()
    
    uci_coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='c_uci')
    uci_coherence = uci_coherence_model.get_coherence()
    
    return umass_coherence, uci_coherence

def print_topics(model, input_file, topic_representations, topic_visual, perplexity, coherence):
    """writes topics to a file"""
    input_filename_without_extension, _ = os.path.splitext(os.path.basename(input_file))
    txt_output_file = f"{model}_{input_filename_without_extension}_topics.txt"
    with open(txt_output_file, "w") as output_file:
        output_file.write("\n\nPerplexity:\n")
        output_file.write(str(perplexity))

        output_file.write("\n\nTopic Coherence Measures:\n")
        output_file.write(f"u_mass: {coherence[0]}\n")
        output_file.write(f"c_uci: {coherence[1]}\n")

        for topic, representation in topic_representations.items():
            output_file.write(f"\n\nTopic {topic}:\n")
            for word, value in representation:
                output_file.write(f"{word}: {round(value, 2)}\n")

    html_output_file = f"{model}_{input_filename_without_extension}_topics_visual.html"
    pio.write_html(topic_visual, html_output_file)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Processing input file...")
    documents = process_input_file(args.input_file)

    logger.info("Fitting and transforming documents...")

    if "swissbert" in args.model:
        topic_model, sentence_model, sentence_tokenizer = set_up_model(args.model, args.language)
        embeddings = np.zeros((len(documents), 768), dtype=np.float32)
        for i, doc in enumerate(documents):
            embedding = generate_swissbert_embedding(doc, sentence_model, sentence_tokenizer)
            embeddings[i, :] = embedding.flatten()
        topics, probabilities = topic_model.fit_transform(documents, embeddings)
    
    elif args.model == "sentence-bert":
        topic_model = set_up_model(args.model, args.language)
        topics, probabilities = topic_model.fit_transform(documents)

    else:
        print("Invalid model name. Please specify a valid model name (sentence-bert, swissbert, sentence-swissbert)")
        exit()
    
    logger.info("Calculating perplexity...")
    log_perplexity = -1 * np.mean(np.log(np.sum(probabilities, axis=1)))
    perplexity = np.exp(log_perplexity)

    logger.info("Calculating topic coherence via UCI and UMASS...")
    umass_coherence, uci_coherence = calculate_coherence(topic_model, topics, documents)
    coherence = (umass_coherence, uci_coherence)

    logger.info("Getting topic representations...")
    topic_representations = {}
    for i in range(topic_model.nr_topics):
        topic_representation = topic_model.get_topic(i)
        if topic_representation:
            topic_representations[i] = topic_representation

    logger.info("Visualizing topics...")
    topic_visual = topic_model.visualize_topics()

    logger.info("Printing topics to files...")
    print_topics(args.model, args.input_file, topic_representations, topic_visual, perplexity, coherence)

if __name__ == "__main__":
    main()
