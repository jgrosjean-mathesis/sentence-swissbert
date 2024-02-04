import os
import glob
import json
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# load desired swissBERT model
model_name = "ZurichNLP/swissbert" # or "jgrosjean-mathesis/sentence-swissbert"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.set_default_language("de_CH")
print("Language set to de_CH")
model.eval()

def generate_sentence_embedding(sentence):

    # tokenize input sentence
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # take tokenized input and pass it through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # extract sentence embeddings via mean pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask

    return embedding

# set source file path
train_file_path = "define source file path"

# get training data
train_pattern = os.path.join(train_file_path, '*.json')
train_files = glob.glob(train_pattern)

# define category dict and category lists
train_categories_with_content = {}
category_film = ["Film", "TV-Serien", "TV", "Netflix", "Hollywood", "Fernsehen"]
category_corona = ["Corona-Impfstoff", "Corona-Impfung", "Corona-Test", "Maskenpflicht", "Coronavirus", "Pandemie", "Omikron", "Booster-Impfung", "Corona-Fallzahlen", "Coronavirus-Mutation", "Covid-19", "Covid-Zertifikat", "Coronavirus-Mutation", "Corona-Patient"]
category_football = ["Fussball", "European Super League", "Frauenfussball", "Frauenfussball-WM", "Fussball-Nationalteam", "Super League", "Fussball-EM", "Fussball-WM", "Champions League", "Premier League", "Bundesliga"]

# iterate over articles to get data
print("getting training data...")
for train_file in train_files:
    with open(train_file, 'r', encoding='utf-8') as article:
        data = json.load(article)
    
        # check if category is empty
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

def evaluation(test_file_path, model, train_categories_with_embeddings):

    # Adjust language according to test data
    if "TS_DE" in test_file_path:
        model.set_default_language("de_CH")
        print("Language set to de_CH")
    elif "TS_FR" in test_file_path:
        model.set_default_language("fr_CH")
        print("Language set to fr_CH")
    elif "TS_IT" in test_file_path:
        model.set_default_language("it_CH")
        print("Language set to it_CH")
    elif "TS_RM" in test_file_path:
        model.set_default_language("rm_CH")
        print("Language set to rm_CH")

    # get test data
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
            if set(categories).intersection(category_film) and not set(categories).intersection(category_corona) and not set(categories).intersection(category_football):

                # extract id
                id = data.get("id", {})

                # extract content and add to dict with id
                raw_content = data.get("content", {})
                content = "\n".join(raw_content.values())
                test_ids_with_content[id] = content

			    # add id and category to dict
                test_ids_with_category[id] = "film"
			
            # check if category is corona
            if set(categories).intersection(category_corona) and not set(categories).intersection(category_film) and not set(categories).intersection(category_football):
            
                # extract id
                id = data.get("id", {})
            
                # extract content and add to dict with id
                raw_content = data.get("content", {})
                content = "\n".join(raw_content.values())
                test_ids_with_content[id] = content

                # add id and category to dict
                test_ids_with_category[id] = "corona"

            # check if category is football
            if set(categories).intersection(category_football) and not set(categories).intersection(category_film) and not set(categories).intersection(category_corona):
            
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
   
    # set up confusion matrix
    y_true = [true_category for _, true_category in sorted_test_ids_with_category]
    y_pred = [predicted_category for _, predicted_category in sorted_test_ids_with_predicted_category]


    # calculate classification report
    class_report = classification_report(y_true, y_pred, digits=4)

    # print classification report
    print("\nClassification Report for:", test_file_path)
    print(class_report)

# Define test data paths and corresponding category labels
test_data_paths = [specify list of test data paths]

# evaluate for each test data path
for test_path in test_data_paths:
    evaluation(test_path, model, train_categories_with_embeddings)
    print("\n")
