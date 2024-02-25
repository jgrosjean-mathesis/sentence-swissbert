import os
import glob
import json

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model.max_seq_length = 512
model.eval()
print("Max Sequence Length:", model.max_seq_length)

# Set source file paths
summary_file_paths_DE = ["define path(s)"]
content_file_paths_DE = ["define path(s)"]
content_file_paths_FR = ["define path(s)"]
content_file_paths_IT = ["define path(s)"]
content_file_paths_RM = ["define path(s)"]

# define summary and content lists
DE_ids = set()
FR_ids = set()
IT_ids = set()
RM_ids = set()
summaries_DE_DE = {}
summaries_DE_FR = {}
summaries_DE_IT = {}
summaries_DE_RM = {}
contents_DE = {}
contents_FR = {}
contents_IT = {}
contents_RM = {}
index_DE = 0
index_FR = 0
index_IT = 0
index_RM = 0
limit = 499

print("extracting data...")
for file_path in content_file_paths_DE:
    pattern = os.path.join(file_path, '*.json')
    json_files = glob.glob(pattern)

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
                DE_ids.add(id)
                raw_content = data.get("content", {})
                content_values = list(raw_content.values())
                content = "\n".join([str(value) for value in content_values])
                contents_DE[id] = content
                
            index_DE += 1
            if index_DE >= limit:
                break

for file_path in content_file_paths_FR:
    pattern = os.path.join(file_path, '*.json')
    json_files = glob.glob(pattern)

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
                FR_ids.add(id)
                raw_content = data.get("content", {})
                content_values = list(raw_content.values())
                content = "\n".join([str(value) for value in content_values])
                contents_FR[id] = content
                
            index_FR += 1
            if index_FR >= limit:
                break

for file_path in content_file_paths_IT:
    pattern = os.path.join(file_path, '*.json')
    json_files = glob.glob(pattern)

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
                IT_ids.add(id)
                raw_content = data.get("content", {})
                content_values = list(raw_content.values())
                content = "\n".join([str(value) for value in content_values])
                contents_IT[id] = content
                
            index_IT += 1
            if index_IT >= limit:
                break

for file_path in content_file_paths_RM:
    pattern = os.path.join(file_path, '*.json')
    json_files = glob.glob(pattern)

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
                RM_ids.add(id)
                raw_content = data.get("content", {})
                content_values = list(raw_content.values())
                content = "\n".join([str(value) for value in content_values])
                contents_RM[id] = content
                
            index_RM += 1
            if index_RM >= limit:
                break

for file_path in summary_file_paths_DE:
    pattern = os.path.join(file_path, '*.json')
    json_files = glob.glob(pattern)

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
                if id in DE_ids:
                    summary = "\n".join([str(value) for value in raw_summary.values()])
                    summaries_DE_DE[id] = summary
                if id in FR_ids:
                    summary = "\n".join([str(value) for value in raw_summary.values()])
                    summaries_DE_FR[id] = summary
                if id in IT_ids:
                    summary = "\n".join([str(value) for value in raw_summary.values()])
                    summaries_DE_IT[id] = summary
                if id in RM_ids:
                    summary = "\n".join([str(value) for value in raw_summary.values()])
                    summaries_DE_RM[id] = summary

# compute embedding and set up embedding dictionaries
summary_embeddings_DE_DE = {}
summary_embeddings_DE_FR = {}
summary_embeddings_DE_IT = {}
summary_embeddings_DE_RM = {}
content_embeddings_DE = {}
content_embeddings_FR = {}
content_embeddings_IT = {}
content_embeddings_RM = {}

print("computing summary embeddings...")

for id, summary in summaries_DE_DE.items():
    summary_embedding = model.encode(summary, convert_to_tensor=True)
    summary_embeddings_DE_DE[id] = summary_embedding

for id, summary in summaries_DE_FR.items():
    summary_embedding = model.encode(summary, convert_to_tensor=True)
    summary_embeddings_DE_FR[id] = summary_embedding

for id, summary in summaries_DE_IT.items():
    summary_embedding = model.encode(summary, convert_to_tensor=True)
    summary_embeddings_DE_IT[id] = summary_embedding

for id, summary in summaries_DE_RM.items():
    summary_embedding = model.encode(summary, convert_to_tensor=True)
    summary_embeddings_DE_RM[id] = summary_embedding

print("computing content embeddings...")

for id, content in contents_DE.items():
    content_embedding = model.encode(content, convert_to_tensor=True)
    content_embeddings_DE[id] = content_embedding


for id, content in contents_FR.items():
    content_embedding = model.encode(content, convert_to_tensor=True)
    content_embeddings_FR[id] = content_embedding


for id, content in contents_IT.items():
    content_embedding = model.encode(content, convert_to_tensor=True)
    content_embeddings_IT[id] = content_embedding


for id, content in contents_RM.items():
    content_embedding = model.encode(content, convert_to_tensor=True)
    content_embeddings_RM[id] = content_embedding

# look for matches
# look for matches
print("calculating cosine similarities and getting matches...")
predicted_matches_DE = {}

for summary_id, summary_embedding in summary_embeddings_DE_DE.items():
    max_similarity = -1
    predicted_content_id = None
    cosine_score_dict = {}

    for content_id, content_embedding in content_embeddings_DE.items():
        cosine_score = util.cos_sim(summary_embedding, content_embedding)
        cosine_score_dict[content_id] = cosine_score

    # find the highest cosine similarity
    predicted_content_id, max_cosine_score = max(cosine_score_dict.items(), key=lambda x: x[1])

    # assemble predictions in dictionary
    predicted_matches_DE[summary_id] = predicted_content_id

# calculate evaluation scores
print("setting up evaluation...")
correct_matches_DE = 0
wrong_matches_DE = 0


for summary_id, predicted_content_id in predicted_matches_DE.items():

    # check if predictions are correct
    if summary_id == predicted_content_id:
        correct_matches_DE += 1
    else:
        wrong_matches_DE += 1

total_cases_DE = correct_matches_DE + wrong_matches_DE
accuracy_DE = f"{100*(correct_matches_DE / total_cases_DE):.2f}"

# print accuracy score
print("\ntotal:\t\t", total_cases_DE)
print("correct:\t", correct_matches_DE)
print("wrong:\t\t", wrong_matches_DE)
print("\naccuracy DE->DE:\t", accuracy_DE, "%\n")

print("calculating cosine similarities and getting matches...")
predicted_matches_FR = {}

for summary_id, summary_embedding in summary_embeddings_DE_FR.items():
    max_similarity = -1
    predicted_content_id = None
    cosine_score_dict = {}

    for content_id, content_embedding in content_embeddings_FR.items():
        cosine_score = util.cos_sim(summary_embedding, content_embedding)
        cosine_score_dict[content_id] = cosine_score

    # find the highest cosine similarity
    predicted_content_id, max_cosine_score = max(cosine_score_dict.items(), key=lambda x: x[1])

    # assemble predictions in dictionary
    predicted_matches_FR[summary_id] = predicted_content_id

# calculate evaluation scores
print("setting up evaluation...")
correct_matches_FR = 0
wrong_matches_FR = 0


for summary_id, predicted_content_id in predicted_matches_FR.items():

    # check if predictions are correct
    if summary_id == predicted_content_id:
        correct_matches_FR += 1
    else:
        wrong_matches_FR += 1

total_cases_FR = correct_matches_FR + wrong_matches_FR
accuracy_FR = f"{100*(correct_matches_FR / total_cases_FR):.2f}"

# print accuracy score
print("\ntotal:\t\t", total_cases_FR)
print("correct:\t", correct_matches_FR)
print("wrong:\t\t", wrong_matches_FR)
print("\naccuracy DE->FR:\t", accuracy_FR, "%\n")

# look for matches
print("calculating cosine similarities and getting matches...")
predicted_matches_IT = {}

for summary_id, summary_embedding in summary_embeddings_DE_IT.items():
    max_similarity = -1
    predicted_content_id = None
    cosine_score_dict = {}

    for content_id, content_embedding in content_embeddings_IT.items():
        cosine_score = util.cos_sim(summary_embedding, content_embedding)
        cosine_score_dict[content_id] = cosine_score

    # find the highest cosine similarity
    predicted_content_id, max_cosine_score = max(cosine_score_dict.items(), key=lambda x: x[1])

    # assemble predictions in dictionary
    predicted_matches_IT[summary_id] = predicted_content_id

# calculate evaluation scores
print("setting up evaluation...")
correct_matches_IT = 0
wrong_matches_IT = 0


for summary_id, predicted_content_id in predicted_matches_IT.items():

    # check if predictions are correct
    if summary_id == predicted_content_id:
        correct_matches_IT += 1
    else:
        wrong_matches_IT += 1

total_cases_IT = correct_matches_IT + wrong_matches_IT
accuracy_IT = f"{100*(correct_matches_IT / total_cases_IT):.2f}"

# print accuracy score
print("\ntotal:\t\t", total_cases_IT)
print("correct:\t", correct_matches_IT)
print("wrong:\t\t", wrong_matches_IT)
print("\naccuracy DE->IT:\t", accuracy_IT, "%\n")

# look for matches
print("calculating cosine similarities and getting matches...")
predicted_matches_RM = {}

for summary_id, summary_embedding in summary_embeddings_DE_RM.items():
    max_similarity = -1
    predicted_content_id = None
    cosine_score_dict = {}

    for content_id, content_embedding in content_embeddings_RM.items():
        cosine_score = util.cos_sim(summary_embedding, content_embedding)
        cosine_score_dict[content_id] = cosine_score

    # find the highest cosine similarity
    predicted_content_id, max_cosine_score = max(cosine_score_dict.items(), key=lambda x: x[1])

    # assemble predictions in dictionary
    predicted_matches_RM[summary_id] = predicted_content_id

# calculate evaluation scores
print("setting up evaluation...")
correct_matches_RM = 0
wrong_matches_RM = 0


for summary_id, predicted_content_id in predicted_matches_RM.items():

    # check if predictions are correct
    if summary_id == predicted_content_id:
        correct_matches_RM += 1
    else:
        wrong_matches_RM += 1

total_cases_RM = correct_matches_RM + wrong_matches_RM
accuracy_RM = f"{100*(correct_matches_RM / total_cases_RM):.2f}"

# print accuracy score
print("\ntotal:\t\t", total_cases_RM)
print("correct:\t", correct_matches_RM)
print("wrong:\t\t", wrong_matches_RM)
print("\naccuracy DE->RM:\t", accuracy_RM, "%\n")
