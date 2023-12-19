import os
import glob
import json
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load swissBERT model
model_name="ZurichNLP/swissbert"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.set_default_language("it_CH")

def generate_sentence_embedding(sentence):

    # Tokenize input sentence
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Set the model to evaluation mode
    model.eval()

    # Take tokenized input and pass it through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract average sentence embeddings from the last hidden layer
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding

# Set source file path
file_paths = ["specify source file path"]

# define summary and content lists
ids = []
summaries = {}
contents = {}

index = 0
limit = 1000

print("extracting data...")
for file_path in file_paths:
    pattern = os.path.join(file_path, '*.json')
    json_files = glob.glob(pattern)

    # Exclude files that start with '2022'
    json_files = [f for f in json_files if not os.path.basename(f).startswith('2022')]

    # iterate over articles
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as article:
            data = json.load(article)

            # check if summary is empty
            raw_summary = data.get("summary", {})
            if not raw_summary:
                continue

            # get summary and content as strings and add them to lists
            else:
                id = data.get("id", {})
                raw_content = data.get("content", {})
                content_values = list(raw_content.values())
                content = "\n".join([str(value) for value in content_values])
                contents[id] = content
                
                summary = "\n".join([str(value) for value in raw_summary.values()])
                summaries[id] = summary

            index += 1
            if index >= limit:
                break

# compute embedding and set up embedding dictionaries
summary_embeddings = {}
content_embeddings = {}

print("computing summary embeddings...")
for id, summary in summaries.items():
    summary_embedding = generate_sentence_embedding(summary)
    summary_embeddings[id] = summary_embedding

print("computing content embeddings...")
for id, content in contents.items():
    content_embedding = generate_sentence_embedding(content)
    content_embeddings[id] = content_embedding

# look for matches
print("calculating cosine similarities and getting matches...")
predicted_matches = {}

for summary_id, summary_embedding in summary_embeddings.items():
    max_similarity = -1
    predicted_content_id = None
    cosine_score_dict = {}

    for content_id, content_embedding in content_embeddings.items():
        cosine_score = cosine_similarity(summary_embedding, content_embedding)
        cosine_score_dict[content_id] = torch.from_numpy(cosine_score)

    # find the highest cosine similarity
    predicted_content_id, max_cosine_score = max(cosine_score_dict.items(), key=lambda x: x[1])

    # assemble predictions in dictionary
    predicted_matches[summary_id] = predicted_content_id

# calculate evaluation scores
print("setting up evaluation...")
correct_matches = 0
wrong_matches = 0


for summary_id, predicted_content_id in predicted_matches.items():

    # check if predictions are correct
    if summary_id == predicted_content_id:
        correct_matches += 1
    else:
        wrong_matches += 1

total_cases = correct_matches + wrong_matches
accuracy = f"{100*(correct_matches / total_cases):.2f}"

# print accuracy score
print("\ntotal:\t\t", total_cases)
print("correct:\t", correct_matches)
print("wrong:\t\t", wrong_matches)
print("\naccuracy:\t", accuracy, "%\n")
