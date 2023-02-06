import streamlit as st

import psycopg2
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="docker"
)


def update_db(name, climate, culture, cuisine, adventure, natural, budget, language, safety):
    with conn:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO ratings (name,climate,culture,cuisine,adventure_activities,natural_beauty,budget,language,safety) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (name, climate, culture, cuisine, adventure, natural, budget, language, safety))
            cur.close()
def update_answers(name, country):
    countries = {'iceland':0, 'maldives':0, 'monaco':0, 'singapore':0, 'egypt':0}
    countries[country.lower()] = 1
    iceland, maldives, monaco, singapore, egypt = countries['iceland'], countries['maldives'], countries['monaco'], countries['singapore'], countries['egypt']
    with conn:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO answers (name, iceland, maldives, monaco, singapore, egypt) VALUES (%s,%s,%s,%s,%s,%s)", (name, iceland, maldives, monaco, singapore, egypt))
            cur.close()

def main():
    if ('name' not in st.session_state) & ('country' not in st.session_state):
        username_form()


def success(name):
    country = st.session_state.country
    update_answers(name,country)
    st.success(f'Great choice, {name}! I knew you would choose {country}..', icon="✅")


def select_countries():

    update_db(st.session_state.name,st.session_state.climate,st.session_state.culture,st.session_state.cuisine,st.session_state.adventure,st.session_state.natural,st.session_state.budget,st.session_state.language,st.session_state.safety)
    with st.form(key="country", clear_on_submit=True):
        option = st.selectbox(
            'Which country would you choose for your trip?',
            ('Iceland', 'Maldives', 'Monaco', 'Singapore', 'Egypt'), key='country')

        submit = st.form_submit_button(
                    "Submit", on_click=success, args=(st.session_state.name,))

def username_form():
    with st.form(key="test"):

        st.header('Rate how important each of the following factors are to you on a scale of 1 to 10 when choosing a destination for your next trip')
        name = st.text_input('Input your name', key="name")


        col1, col2, col3, col4 = st.columns(4)
        with col1:
            climate = st.slider('Climate', 0, 10, key="climate")

        with col2:
            culture = st.slider('Culture', 0, 10, key="culture")
        with col3:
            cuisine = st.slider('Cuisine', 0, 10, key="cuisine")
        with col4:
            adventure = st.slider('Adventure activities', 0, 10, key="adventure")


        col5, col6,col7,col8 = st.columns(4)

        with col5:
            natural = st.slider('Natural beauty', 0, 10, key="natural")
        with col6:
            budget = st.slider('Budget', 0, 10,key="budget")
        with col7:
            language = st.slider('Language', 0, 10,key="language")
        with col8:
            safety = st.slider('Safety', 0, 10,key="safety")


        st.form_submit_button("Submit", on_click=select_countries)

if __name__ == "__main__":
    main()
