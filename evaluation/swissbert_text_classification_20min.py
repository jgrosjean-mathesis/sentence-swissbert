import os
import glob
import json
from sklearn.metrics import classification_report

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
model.max_seq_length = 512
model.eval()

# set source file path
train_file_path = "specify source file path"

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
    train_categories_with_embeddings[model.encode(train_content, convert_to_tensor=True)] = train_category

def evaluation(test_file_path, train_categories_with_embeddings):

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
        test_ids_with_embeddings[test_id] = model.encode(test_content, convert_to_tensor=True)

    # predict category for test data and calculate accuracy
    print("predicting categories via cosine similarity...")
    test_ids_with_predicted_category = {}

    for test_id, test_embedding in test_ids_with_embeddings.items():
        max_similarity = -1
        predicted_category = None
        cosine_score_dict = {}

        # calculate cosine similarity scores for test content and all training embeddings
        for train_embedding, train_category in train_categories_with_embeddings.items():
            cosine_score = util.cos_sim(test_embedding, train_embedding)
            cosine_score_dict[cosine_score] = train_category

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

# define test data paths and category labels
test_data_paths = [specify list of test data paths]

# evaluate for each test data path
for test_path in test_data_paths:
    evaluation(test_path, train_categories_with_embeddings)
    print("\n")
