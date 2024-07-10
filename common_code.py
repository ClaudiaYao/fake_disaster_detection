from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer #Used to lemmatize words
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import nltk

# the following error happens on mac
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# the above is not needed on Windows

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
tw_tokenizer = TweetTokenizer(reduce_len=True)
nlp = spacy.load("en_core_web_sm")

def remove_stopwords_lemmatizer(text):        
    sent = []
    # doc = nlp(text)
    
    for token in tw_tokenizer.tokenize(text):
        if token not in stop_words:
            sent.append(token)
    return " ".join(sent)

def customize_stopwords():
  fillerWord = ("so","yeah","okay","um","uh","mmm","ahan","uh","huh","ahm","oh","sooo","uh","huh","yeh","yah","hmm","bye")
  stop_words.add(fillerWord)

  keep_words =["not","nor","neither", "no"]
  for i in keep_words:
      stop_words.discard(i)
  
def clean_text(text):

    text = str(text).lower()
    # replace abbreviation correct misspelling words
    text = preprocess_chat_text(text)

    text = re.sub(r"http://t.co/\S+", " LINK ", str(text))
    text = re.sub(r"@\w+", " ", text)  # Remove placeholder [+XYZ chars] 
    text = re.sub(r"\[\w+\]", " ", text)  # Remove placeholder [+XYZ chars] 
    text = re.sub(r"[(/)*-]", " ", text) 
    text = re.sub(r"[\.]{2,}", " ", text)  # Remove ellipsis
    text = re.sub(r"\d+", " ", text) 
    # text = re.sub(r"[#@]", " ", text) # remove hashtag signs
    text = re.sub(r"yo{1,}u{1,}", "you", text)
    text = re.sub(r"ah{1,}", " relief, happy ", text)
    text = re.sub(r"alaugh\w+", "laugh", text)
    text = re.sub(r"angry{1,}", "angry", text)
    text = re.sub(r"boo{1,}m", "boom", text)
    text = re.sub(r"cherr{1,}yy{1,}", "cherry", text)
    text = re.sub(r"class{1,}y", "classy", text)
    text = re.sub(r"cu{2,}te", "cute", text)
    text = re.sub(r"fu{2,}ck\w+", "fuck", text)
    text = re.sub(r"him{2,}", "him", text)
    text = re.sub(r"(no){2,}yes", " finally ok ", text)
    text = re.sub(r"yee{1,}ss{1,}", "yes", text)
    text = re.sub(r"(no){2,}", "no", text)
    text = re.sub(r"agaa{1,}in", "again", text)
    text = re.sub(r"a{1,}g{0,}h{1,}", "angry", text)
    text = re.sub(r"a{1,}laug{0,}h{1,}", "laugh", text)
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content

    # tokens = tw_tokenizer.tokenize(text)
    # text = ""
    text = remove_stopwords_lemmatizer(text)
    return text

def preprocess_chat_text(text):
    # Expand common abbreviations
    abbreviation_mapping = {
        ":d": "laugh",
        "lol": "laugh out loud",
        "brb": "be right back",
        "omg": "oh my god",
        "sec": "second",
        'idk': "i do not know",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "isn't": "is not",
        "hadn't": "had not",
        "haven't": "have not",
        "mightn't": "might not",
        "needn't": "need not",
        "wasn't": "was not",
        "weren't": "were not",
        "won't": "will not",
        "shouldn't": "should not",
        "lmao": "laughing my ass off",
        "'ve": " have",
        "I'm": "I am",
        "'s": " is",
        "can't": "can not",
        "it's": "it is",
        "you're": "you are",
        "nvm": "never mind",
        "haha": "laugh",
        "it'll": "it will",
        "'ll": " will",
        "r/": " are ",
        # Add more mappings as needed
    }
    
    # Replace abbreviations with their expanded forms
    for abbreviation, expansion in abbreviation_mapping.items():
        text = text.replace(abbreviation, expansion)
    
    # Normalize common misspellings
    misspelling_mapping = {
        "u": "you",
        "gr8": "great",
        # Add more mappings as needed
    }

    emonicons_mapping = {
       ":)": "happy",
       ":-)": "happy",
       ";)" : "happy winking",
        ";-)": "happy winking",
        ":P": "sticking out tongue",
        ":-P": "sticking out tongue",
        ":D" :	"open-mouthed grin",
        ":-D": "open-mouthed grin",
        ":(":	"unhappy",
        ":-(": "unhappy",
        ":~(": "crying",
        ":-|": "unemotional",
        ">:-(": "very unhappy",
        "8-)": "wide eyed happyness",
        ":-O": "surprise",
        ":o": "surprise",
        "8-O": "wide-eyed shouting",
        ">8-O": "mad wide-eyed shouting",
        "|-|": "asleep",
        "==|:-)": "silly",
    }
    
    # Replace misspelled words with their correct forms
    for misspelling, correction in misspelling_mapping.items():
        text = re.sub(r"\b{}\b".format(misspelling), correction, text)
    
    # Replace emoticons with their mapping words
    for emoticon, correction in emonicons_mapping.items():
        text = text.replace(emoticon, correction)
    return text

def clean_text_wrapper(df_train, df_test):
    customize_stopwords()

    print("Start cleaning...")
    df_train['cleaned_text'] = df_train.text.apply(clean_text)
    df_test['cleaned_text'] = df_test.text.apply(clean_text)
    print("done!")
    # df_train.to_csv("data/cleaned_train.csv", index=False)
    # df_test.to_csv("data/cleaned_test.csv", index=False)
    return df_train, df_test

def encode_label(df_train):
    le = LabelEncoder()
    df_train["upt_label"] = le.fit_transform(df_train['Label'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    le_label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print("name maps to numeric label: ", le_name_mapping)
    print("numeric label maps to name: ", le_label_mapping)
    return le_name_mapping, le_label_mapping, df_train["upt_label"]

def encode_doc(para_max_features, para_ngram_range, train_docs, test_docs):
    vectorizer = TfidfVectorizer(max_features=para_max_features, ngram_range = para_ngram_range,  min_df=1, max_df=0.7) 
    text_train = vectorizer.fit_transform(train_docs)
    text_test = vectorizer.transform(test_docs)
    tfidf_tokens = vectorizer.get_feature_names_out()
    print("text_train.shape:", text_train.shape)
    print("The first 100 tokens:", tfidf_tokens[:100])
    return text_train, text_test

def draw_piechart(df_train):
    # show pie chart for the training data distribution
    kwargs = dict(
        startangle = 90,
        autopct = '%1.1f%%',
        labels = df_train['Label'].value_counts().index,
        colors=("#048a81", "#06d6a0", "#54C6EB", "#8A89C0", "#CDA2AB"),
        radius = 0.9, 
        textprops={'size': 'medium'},
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    fig.subplots_adjust(top=0.7, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.axis('equal')
    ax.margins(0, 0)
    ax.pie(df_train['Label'].value_counts(), **kwargs)
    return fig

def draw_confusion_matrix(cm, label_names):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(top=0.7, bottom=0, right=0.9, left=0, hspace=0, wspace=0)

    ax.set_title("Confusion Matrix", size = 12)
    ConfusionMatrixDisplay(
        cm, 
        display_labels =label_names).plot(
        include_values = True, 
        cmap="Blues", 
        ax=ax, 
        colorbar=False)
    ax.tick_params(axis='x', which='both', bottom = False, top = False, labelbottom = True,  labelsize=9)
    return fig

