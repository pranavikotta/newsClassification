import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from transformers import BertTokenizer, BertModel

# obtaining datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(train_df.head())
print(test_df.head())

X_train_titles = train_df['Title']
X_train_desc = train_df['Description']
y_train = train_df['Class Index']
# 1 - world , 2 - sports , 3 - business , 4 - science/technology
X_test_titles = test_df['Title']
X_test_desc = test_df['Description']
y_test = test_df['Class Index']

tdidf_title = TfidfVectorizer(lowercase=True, stop_words='english')
tdidf_desc = TfidfVectorizer(lowercase=True, stop_words='english')

titles_trainvector = tdidf_title.fit_transform(X_train_titles)
desc_trainvector = tdidf_desc.fit_transform(X_train_desc)
titles_testvector = tdidf_title.transform(X_test_titles)
desc_testvector = tdidf_desc.transform(X_test_desc)

X_train_vector = hstack([titles_trainvector, desc_trainvector])
X_test_vector = hstack([titles_testvector, desc_testvector])

model_MNB = MultinomialNB()
model_MNB.fit(X_train_vector, y_train)
y_pred = model_MNB.predict(X_test_vector)

print("Classification Report: ")
print(classification_report(y_test, y_pred))

# # implementing BERT transformer
# tokenizer = BertTokenizer.from_pretrained('bert-based-uncased')
# # prepare lists of strings to pass to tokenizer
# train_titles = X_train_titles.tolist
# train_desc = X_train_desc.tolist
# test_titles = X_test_titles.tolist
# test_desc = X_test_desc.tolist

# # tokenize as sentence pairs; titles as segment a and descriptions as segment b
# train_encoder = tokenizer{
#     train_titles,
#     train_desc,
#     truncation = True,
#     padding = True,
#     max_length = 128,
#     return_tensors = 'pt'
# }
