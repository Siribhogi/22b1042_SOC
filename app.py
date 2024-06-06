import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
import streamlit as st

def train_model():
    with open('data.txt', 'r') as file:
        text=file.read()

    tokenizer=Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words=len(tokenizer.word_index)+1

    input_sequences=[]
    for line in text.split('.'):
        token_list=tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            n_gram_sequence=token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len=max([len(seq) for seq in input_sequences])
    input_sequences=np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X=input_sequences[:,:-1] 
    Y=input_sequences[:,-1]

    Y=to_categorical(Y,num_classes=total_words)

    model=Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(200))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X,Y,epochs=150)

    model.save('next_word_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('max_sequence_len.pickle', 'wb') as handle:
        pickle.dump(max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_and_tokenizer():
    model=tf.keras.models.load_model('next_word_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('max_sequence_len.pickle', 'rb') as handle:
        max_sequence_len = pickle.load(handle)
    return model, tokenizer, max_sequence_len

def predict_next_words(model, tokenizer, text, num_predict, max_sequence_len):
    for _ in range(num_predict):
        token_list=tokenizer.texts_to_sequences([text])[0]
        token_list=pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted=model.predict(token_list, verbose=0)
        predicted_word_index=np.argmax(predicted, axis=1)[0]
        output_word=tokenizer.index_word[predicted_word_index]
        text+=' '+output_word
    return text

def main():
    st.title('Next Word Prediction')
    input_text=st.text_input('Enter a starting text:')
    num_predict=st.number_input('Number of words to predict:', min_value=1, max_value=8, value=5, step=1)

    if st.button('Predict'):
        model, tokenizer, max_sequence_len=load_model_and_tokenizer()
        result=predict_next_words(model, tokenizer, input_text, num_predict, max_sequence_len)
        st.write(result)

if __name__ == "__main__":
    import argparse

    parser=argparse.ArgumentParser(description='Train model or run the web app')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--app', action='store_true', help='Run the web app')

    args=parser.parse_args()

    if args.train:
        train_model()
    elif args.app:
        main()
    else:
        print("Please specify --train to train the model or --app to run the web app")