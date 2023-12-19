import os
import glob
import json
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load swissBERT model
model_name = "ZurichNLP/swissbert"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.set_default_language("de_CH")

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
train_file_path = "specify train file path"
test_file_path = "specify test file path"

# Get training data
train_pattern = os.path.join(train_file_path, '*.json')
train_files = glob.glob(train_pattern)

# define category dict and category lists
train_categories_with_content = {}
category_film = ["Film", "TV-Serien"]
category_corona = ["Corona-Impfung", "Coronavirus", "Pandemie", "Omikron", "Coronavirus-Mutation"]
category_football = ["Fussball", "Fussball-Nationalteam"]

# iterate over articles to get data
print("getting training data...")
for train_file in train_files:
    with open(train_file, 'r', encoding='utf-8') as article:
        data = json.load(article)
    
        # check if category or content is empty
        categories = data.get("category", None)
        if not categories:
            continue

        # check if category is a single string and, if true, turn into list
        elif isinstance(categories, str):
            categories = [categories]

        

        # check if category is film
        if set(categories).intersection(category_film) and not set(categories).intersection(category_corona) and not set(categories).intersection(category_football):

            # extract content
            raw_content = data.get("content", {})
            content = "\n".join(raw_content.values())

            # add category and content to train category dict
            train_categories_with_content[content] = "film"

        # check if category is corona
        if set(categories).intersection(category_corona) and not set(categories).intersection(category_film) and not set(categories).intersection(category_football):

            # extract content
            raw_content = data.get("content", {})
            content = "\n".join(raw_content.values())

            # add category and content to train category dict
            train_categories_with_content[content] = "corona"

        # check if category is football
        if set(categories).intersection(category_football) and not set(categories).intersection(category_film) and not set(categories).intersection(category_corona):

            # extract content
            raw_content = data.get("content", {})
            content = "\n".join(raw_content.values())

            # add category and content to train category dict
            train_categories_with_content[content] = "football"

# compute embeddings for content and store them in dict
print("computing training embeddings...")
train_categories_with_embeddings = {}

for train_content, train_category in train_categories_with_content.items():
    train_categories_with_embeddings[generate_sentence_embedding(train_content)] = train_category

# Get test data
test_pattern = os.path.join(test_file_path, '*.json')
test_files = glob.glob(test_pattern)

# define content list, id list and true test category dict with key=id and value=true category
test_ids_with_category = {}
test_ids_with_content = {}

# get test data
print("getting test data...")
for test_file in test_files:
    with open(test_file, 'r', encoding='utf-8') as article:
        data = json.load(article)
        
        # check if category is empty
        categories = data.get("category", [])
        if not categories:
            continue

        # check if category is a single string
        elif isinstance(categories, str):
            categories = [categories]  # convert single string to list


        # check if category is film
        if set(categories).intersection(category_film):

            # extract id
            id = data.get("id", {})

            # extract content and add to dict with id
            raw_content = data.get("content", {})
            content = "\n".join(raw_content.values())
            test_ids_with_content[id] = content

			# add id and category to dict
            test_ids_with_category[id] = "film"
			
        # check if category is corona
        if set(categories).intersection(category_corona):
            
            # extract id
            id = data.get("id", {})
            
             # extract content and add to dict with id
            raw_content = data.get("content", {})
            content = "\n".join(raw_content.values())
            test_ids_with_content[id] = content

            # add id and category to dict
            test_ids_with_category[id] = "corona"

        # check if category is football
        if set(categories).intersection(category_football):
            
            # extract id
            id = data.get("id", {})
            
            # extract content and add to dict with id
            raw_content = data.get("content", {})
            content = "\n".join(raw_content.values())
            test_ids_with_content[id] = content
            
            # add id and category to dict
            test_ids_with_category[id] = "football"

# set up test embeddings
print("calculating test embeddings...")
test_ids_with_embeddings = {}

for test_id, test_content in test_ids_with_content.items():
    test_ids_with_embeddings[test_id] = generate_sentence_embedding(test_content)

# predict category for test data and calculate accuracy
print("predicting categories via cosine similarity...")
test_ids_with_predicted_category = {}

for test_id, test_embedding in test_ids_with_embeddings.items():
    max_similarity = -1
    predicted_category = None
    cosine_score_dict = {}

    # calculate cosine similarity scores for test content and all training embeddings
    for train_embedding, train_category in train_categories_with_embeddings.items():
        cosine_score = cosine_similarity(test_embedding, train_embedding)
        cosine_score_dict[torch.from_numpy(cosine_score)] = train_category

    # find the highest cosine similarity
    max_cosine_score = max(cosine_score_dict.keys())
    predicted_category = cosine_score_dict[max_cosine_score]

    # append the predicted category to the dictionary with id as key
    test_ids_with_predicted_category[test_id] = predicted_category

# calculate evaluation metrics of model
print("evaluating the output...")
correct_categories = 0
wrong_categories = 0

true_positives_film = 0
false_positives_film = 0
true_negatives_film = 0
false_negatives_film = 0

true_positives_corona = 0
false_positives_corona = 0
true_negatives_corona = 0
false_negatives_corona = 0

true_positives_football = 0
false_positives_football = 0
true_negatives_football = 0
false_negatives_football = 0

for test_id, true_category in test_ids_with_category.items():

    # get overall metrics
    if test_ids_with_predicted_category[test_id] == true_category:
        correct_categories += 1

    else:
        wrong_categories += 1
    
    # get film metrics
    if true_category == "film":
        if test_ids_with_predicted_category[test_id] == "film":
            true_positives_film += 1
        else:
            false_negatives_film += 1
    
    if true_category != "film":
        if test_ids_with_predicted_category[test_id] != "film":
            true_negatives_film += 1
        else:
            false_positives_film += 1

    # get corona metrics
    if true_category == "corona":
        if test_ids_with_predicted_category[test_id] == "corona":
            true_positives_corona += 1
        else:
            false_negatives_corona += 1
    
    if true_category != "corona":
        if test_ids_with_predicted_category[test_id] != "corona":
            true_negatives_corona += 1
        else:
            false_positives_corona += 1
    
    # get football metrics
    if true_category == "football":
        if test_ids_with_predicted_category[test_id] == "football":
            true_positives_football += 1
        else:
            false_negatives_football += 1
    
    if true_category != "football":
        if test_ids_with_predicted_category[test_id] != "football":
            true_negatives_football += 1
        else:
            false_positives_football += 1

# print train / test ratio
train_data = len(train_categories_with_content)
test_data = len(test_ids_with_content)
total_data = train_data + test_data

print("\nTrain data ratio is:\t", f"{100*(train_data / total_data):.2f}", "%")
print("Test data ratio is:\t", f"{100*(test_data / total_data):.2f}", "%")

# set up confusion matrix
film_recall = true_positives_film / (true_positives_film + false_negatives_film)
film_precision = true_positives_film / (true_positives_film + false_positives_film)
film_f1_score = (2*film_recall*film_precision)/(film_recall+film_precision)

corona_recall = true_positives_corona / (true_positives_corona + false_negatives_corona)
corona_precision = true_positives_corona / (true_positives_corona + false_positives_corona)
corona_f1_score = (2*corona_recall*corona_precision)/(corona_recall+corona_precision)

football_recall = true_positives_football / (true_positives_football + false_negatives_football)
football_precision = true_positives_football / (true_positives_football + false_positives_football)
football_f1_score = (2*football_recall*football_precision)/(football_recall+football_precision)

total_recall = f"{100*((film_recall + corona_recall + football_recall) / 3):.2f}"
total_precision = f"{100*((film_precision + corona_precision + football_precision) / 3):.2f}"
total_f1_score = f"{100*((film_f1_score + corona_f1_score + football_f1_score) / 3):.2f}"

total_cases = correct_categories + wrong_categories
total_accuracy = f"{100*(correct_categories / total_cases):.2f}"


# print total accuracy score
print("\ntotal:\t\t", total_cases)
print("\nrecall:\t\t", total_recall, "%")
print("precision:\t", total_precision, "%")
print("f1-score:\t", total_f1_score, "%")
print("\ntotal accuracy:\t", total_accuracy, "%")
