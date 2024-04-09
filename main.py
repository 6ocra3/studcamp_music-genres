import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Two model inference")

token = st.secrets["STREAMLIT_TOKEN"]

css = """
<style>
    /* Увеличиваем шрифт лейблов */
    .stSlider label div p {
        font-size: 20px !important;
    }
</style>
"""


st.markdown(css, unsafe_allow_html=True)

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

    st.markdown(css, unsafe_allow_html=True)

    start_year = 2015
    if 'year' not in st.session_state:
        st.session_state.year = start_year

    graph_placeholder = st.empty()

    st.session_state.year = st.slider('Выбери год', 1900, 2022, start_year)

    with graph_placeholder.container():
        genres_by_year(st.session_state.year)

    st.title("Распознование жанра песни")

    model = input_features()
    st.header("Введите текст:")
    lyrics = st.text_area("Текст песни", height=400, label_visibility="hidden")
    inference(model, lyrics)

def genres_by_year(year):
    df = pd.read_csv('songs_by_year_and_genre.csv')

    filtered_data = df[(df['year'] >= year-2) & (df['year'] <= year+2) & (df['tag'] != 'misc')]

    grouped_data = filtered_data.groupby('tag')['count'].sum().reset_index()

    grouped_data = grouped_data.sort_values(by='count', ascending=False)

    labels = grouped_data['tag']
    data = grouped_data['count']

    # Создаем фигуру и ось с равным аспектом
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))

    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    # Создаем диаграмму "donut" с помощью параметра `wedgeprops`
    wedges, texts, autotexts = ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=140,
                                      pctdistance=0.5,
                                      wedgeprops=dict(width=0.3))

    # Настройка дизайна: добавляем заголовок и улучшаем расположение текста
    ax.set_title(f'Доля жанров среди песен за период {year-2}-{year+2} годы', color='white', fontsize=14)

    # Делаем диаграмму более привлекательной, раздвигая текстовые метки
    plt.setp(autotexts, size=11, weight="bold", color="white")
    plt.setp(texts, size=11, weight="bold", color="white")

    st.pyplot(fig)

if __name__ == "__main__":
    show_main_page()
