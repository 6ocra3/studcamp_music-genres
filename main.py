import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tflr
import seaborn as sns

st.set_page_config(page_title="Two model inference")
token = st.secrets["STREAMLIT_TOKEN"]

css = """
<style>
    /* Увеличиваем шрифт лейблов */
    .stSlider label div p {
        font-size: 20px !important;
    }
    h1{
    font-size: 30px !important;
    }
    .stRadio p{
        font-size: 24px !important;
    }
    .stRadio div label div{
    margin: auto
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

API_URL_dict = {
    "tf-idf+logreg": tflr.predict,
    "World2Vec": "",
    "TF-IDF": "",
}

headers = {"Authorization": f"Bearer {token}"}


def query(model, data):
    if API_URL_dict[model]:
        return API_URL_dict[model](data)[0]
        # response = requests.post(API_URL_dict[model], headers=headers, data=data)
        # return response.json()
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

def genres_by_year(year):
    df = pd.read_csv('songs_by_year_and_genre.csv')

    filtered_data = df[(df['year'] >= year - 2) & (df['year'] <= year + 2) & (df['tag'] != 'misc')]

    grouped_data = filtered_data.groupby('tag')['count'].sum().reset_index()

    grouped_data = grouped_data.sort_values(by='count', ascending=False)

    labels = grouped_data['tag']
    data = grouped_data['count']

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))

    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    wedges, texts, autotexts = ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=140,
                                      pctdistance=0.5,
                                      wedgeprops=dict(width=0.3))

    ax.set_title(f'Доля жанров среди песен за период {year - 2}-{year + 2} годы', color='white', fontsize=14)

    plt.setp(autotexts, size=11, weight="bold", color="white")
    plt.setp(texts, size=11, weight="bold", color="white")

    st.pyplot(fig)


def heatmap():
    matrix = [[1., 0.81281475, 0.72420595, 0.75301047, 0.65628327, 0.71672701],
              [0.81281475, 1., 0.85626718, 0.92609532, 0.66257472, 0.87868144],
              [0.72420595, 0.85626718, 1., 0.97197867, 0.80509611, 0.92482678],
              [0.75301047, 0.92609532, 0.97197867, 1., 0.79099902, 0.94956425],
              [0.65628327, 0.66257472, 0.80509611, 0.79099902, 1., 0.7664936],
              [0.71672701, 0.87868144, 0.92482678, 0.94956425, 0.7664936, 1.]]
    genres = ['rap', 'rb', 'rock', 'pop', 'misc', 'country']
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Oranges',
                xticklabels=genres, yticklabels=genres, color="white")

    ax.set_title('Косинусное сходство между жанрами', size=20, color='white')
    ax.set_xlabel('Жанры', size=13, color='white')
    ax.set_ylabel('Жанры', size=13, color='white')


    ax.tick_params(colors='white', size=13, which='both')

    cbar1 = ax.collections[0].colorbar
    cbar1.ax.tick_params(labelsize=13, colors='white')

    st.pyplot(fig)


def page_visualizations():
    """Страница с визуализациями."""

    start_year = 2015
    if 'year' not in st.session_state:
        st.session_state.year = start_year

    graph_placeholder = st.empty()
    st.session_state.year = st.slider('Выбери год', 1900, 2022, start_year)

    with graph_placeholder.container():
        genres_by_year(st.session_state.year)

    heatmap()

def page_model_inference():
    """Страница с селектором модели и вводом текста."""
    st.title("Распознование жанра песни")

    model = input_features()
    st.header("Введите текст:")
    lyrics = st.text_area("Текст песни", height=400, label_visibility="hidden")
    inference(model, lyrics)

# Функция main теперь отвечает за управление навигацией
def main():
    tflr.init()

    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите страницу:", ["Визуализации", "Распознавание жанра"])

    if page == "Визуализации":
        page_visualizations()
    elif page == "Распознавание жанра":
        page_model_inference()

if __name__ == "__main__":
    main()
