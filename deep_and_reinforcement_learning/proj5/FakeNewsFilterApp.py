
import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from tensorflow import keras 
import re 

@st.cache(allow_output_mutation=True)
def loadmodel():
    download('stopwords')
    download('punkt')
    download('wordnet')
    download('omw-1.4')
    with open('deep_and_reinforcement_learning/proj5/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    loaded_model = keras.models.load_model("deep_and_reinforcement_learning/proj5/final_model_tf")
    return loaded_model, tokenizer

def CorpusGen(text):
    # Removes numbers, lowercases words, removes stopwords, word tokenizes, and finally lemmantizes
    lemmatizer = WordNetLemmatizer()
    message = re.sub('[^a-zA-Z]', ' ', text)
    message = message.lower()
    message = word_tokenize(message)
    message = [lemmatizer.lemmatize(w) for w in message if not w in stopwords.words('english')]
    message = ' '.join(message)
    return message

def tokenize(tokenizer, corpus):
    sequence = tokenizer.texts_to_sequences([corpus])
    padded_text = keras.utils.pad_sequences(sequence, maxlen=600, padding='post', truncating='post')
    return padded_text

def testmodel(loaded_model, tokenizer, text):
    if(len(text) > 100):
        with st.spinner('Analyzing...'):
            corpus = CorpusGen(text)
            padded_text = tokenize(tokenizer, corpus)
            prediction = loaded_model.predict(padded_text)
            if(prediction[0] < 0.20):
                if(prediction[0] < 0.02):
                    result.text("FAKE NEWS! With high likelihood")
                elif(prediction[0] < 0.10):
                    result.text("Fake News! With fair likelihood")
                else:
                    result.text("Elements of Fake News")
            elif(prediction[0] > 0.80):
                if(prediction[0] > 0.99):
                    result.text("Real News, with high likelihood")
                elif(prediction[0] > 0.90):
                    result.text("Real News, with fair likelihood")
                else:
                    result.text("Elements of Real News")
            else:
                result.text("Article Analysis was Inconclusive")
    else:
        result.text('Input Valid Article with Word Count Greater than 100')
            
def clear_text():
    st.session_state["title"] =""
    st.session_state["text"] = ""
    st.session_state["result"] = ""



# UI of the App

loaded_model, tokenizer = loadmodel()

st.title('Fake vs Real News Classifier')
st.subheader(" ")

st.subheader("Input Article: ")

input_title = st.text_input('Article Title', key='title')

input_text = st.text_area('Article Text', key='text')

result = st.empty()

run_model = st.button("Analyze")
if(run_model):
    testmodel(loaded_model, tokenizer, input_text)

click_clear = st.button('Clear', on_click=clear_text)

    