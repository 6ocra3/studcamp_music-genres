import streamlit as st
import requests
import os

token = os.getenv('STREAMLIT_TOKEN')

API_URL_dict = {
    "Bag of words": "",
    "World2Vec": "",
    "TF-IDF": "",
}

headers = {"Authorization": f"Bearer {token}"}


def query(model, data):
    if API_URL_dict[model]:
        response = requests.post(API_URL_dict[model], headers=headers, data=data)
        return response.json()
    else:
        return "in progress"


def input_features():
    model = st.selectbox(
        "Модель: ", API_URL_dict.keys()
    )
    return model


def predict(model, data):
    result = query(model, data)
    return result


def inference(model, lyrics):
    if lyrics:
        output = predict(model, lyrics)
        st.header("Распознанный жанр:")
        st.subheader(output)


def show_main_page():
    st.set_page_config(
        page_title="Two model inference"
    )
    st.title("Распознование жанра песни")

    model = input_features()
    st.header("Введите текст:")
    lyrics = st.text_area("Текст песни", height=400, label_visibility="hidden")
    inference(model, lyrics)


if __name__ == "__main__":
    show_main_page()
