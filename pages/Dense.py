import streamlit as st
import pandas as pd
import psycopg2
import graphviz

import numpy as np
from numpy.random import randn

from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

def highlight(data):
    is_min = data == data.nsmallest(1).iloc[-1]
    is_max = data == data.nlargest(1).iloc[0]
    styles = [''] * len(data)
    min_index = np.flatnonzero(is_min.to_numpy())
    max_index = np.flatnonzero(is_max.to_numpy())
    for i in min_index:
        styles[i] = 'background-color: rgba(255,0,0,0.3)'
    for i in max_index:
        styles[i] = 'background-color: rgba(0,255,0,0.3)'
    return styles


graph_loss = graphviz.Digraph(graph_attr={'rankdir':'LR', 'TBbalance':"max"})
graph_loss.edge('X', 'Mult')
graph_loss.edge('W1', 'Mult')
graph_loss.edge('Mult', 'Relu')
graph_loss.edge('Relu', 'Prediction')
graph_loss.edge('W2', 'Prediction')
graph_loss.edge('Prediction', 'L2 Loss', label='(y_pred -y_true)**2')
st.graphviz_chart(graph_loss)



def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


# Define your project id and dataset id
project_id = 'wagon-bootcamp-355610'
dataset_id = 'ml'

# Query to get data from the "ratings" table
query_ratings = f"SELECT * FROM `{project_id}.{dataset_id}.ratings`"
# Execute the query and load the result into a pandas DataFrame
x = pd.read_gbq(query_ratings, credentials=credentials)

# Query to get data from the "answers" table
query_answers = f"SELECT * FROM `{project_id}.{dataset_id}.answers`"
# Execute the query and load the result into a pandas DataFrame
y = pd.read_gbq(query_answers, credentials=credentials)



expander_X = st.expander("X")
expander_X.dataframe(x)
expander_Y = st.expander("Y (y_true)")
expander_Y.dataframe(y)

X = x.drop('name', axis=1)


Y = y.drop('name', axis=1)
N, D_in, H, D_out = X.shape[0], X.shape[1], 10, Y.shape[1]

X_matrix = X.astype('float').to_numpy()
Y_matrix = Y.astype('float').to_numpy()

mean = np.mean(X_matrix)
std = np.std(X_matrix)
X_matrix = (X_matrix-mean)/std
expander_X_norm = st.expander("X Normalized")
X_matrix_df = pd.DataFrame(X_matrix)
X_matrix_df.columns = X.columns
X_matrix_df.index = x.name
expander_X_norm.dataframe(X_matrix_df)



if 'count' not in st.session_state:
	st.session_state.count = 0

def increment_counter():
	st.session_state.count += 1

col1, col2, col3  = st.columns(3)

with col1:
    initialize =  st.button('Epoch iteration',on_click=increment_counter)
with col2:
    if st.button('Clear Cache'):
        st.runtime.legacy_caching.clear_cache()


learning_rate = st.number_input('Learning Rate', value=0.009,key='learning_rate',format="%.3f")




@st.cache(allow_output_mutation=True)
def get_my_list():
    return []

if initialize:
    if st.session_state.count ==1:
        w1, w2 = randn(D_in, H), randn(H, D_out)
    else:
        w1,w2 = st.session_state.w1, st.session_state.w2

    st.write('Iteration = ', st.session_state.count)
    st.header(':green[Forward Pass]')

    w1 = pd.DataFrame(w1)
    w1.columns = [f'W1_feature {i}' for i in range(1, w1.shape[1]+1)]
    w1.index = [X.columns]


    w2 = pd.DataFrame(w2)
    w2.columns = Y.columns
    w2.index = [w1.columns]


    expander_W1 = st.expander("W1 with 10 layers")
    expander_W1.dataframe(w1.style.apply(highlight, axis=1))

    expander_W2 = st.expander("W2")
    expander_W2.dataframe(w2.style.apply(highlight, axis=1))


    with st.expander("X * W1"):
        st.header("X * W1 | x.dot(w1)")
        w1_mult_x = X_matrix.dot(w1)
        w1_mult_x_style = pd.DataFrame(w1_mult_x)
        w1_mult_x_style.columns = w1.columns
        w1_mult_x_style.index = x.name
        st.dataframe(w1_mult_x_style)


    with st.expander("Relu(0,x)"):
        st.header("Relu(0,x) | np.maximum(0, w1_mult_x)")
        h = np.maximum(0, w1_mult_x)
        h_style = pd.DataFrame(h)
        h_style.columns = w1.columns
        h_style.index = x.name
        st.dataframe(h_style.style.applymap(lambda x: 'background-color: rgba(255,0,0,0.3)' if x == 0 else ''))


   # Forward pass:
    # Relu*W2 | h.dot(w2)
    with st.expander("Y_Pred = Relu*W2"):
        st.header("Y_Pred = Relu*W2 | np.exp(h.dot(w2)) / np.sum(np.exp(h.dot(w2)))")
        y_pred_raw = h.dot(w2)
        y_pred = np.exp(y_pred_raw) / np.sum(np.exp(y_pred_raw))  # softmax
        y_pred_style = pd.DataFrame(y_pred)
        y_pred_style.columns = Y.columns
        y_pred_style.index = x.name
        st.dataframe(y_pred_style.style.apply(highlight, axis=1))

    # Loss calculation
    st.subheader('Loss - (-np.sum(y_true * np.log(y_pred)))')
    loss = -np.sum(Y_matrix * np.log(y_pred))  # categorical cross-entropy
    st.success(np.round(loss,2))
    loss_list = get_my_list()
    loss_list.append(loss)

    # Backprop calculation
    st.header(':red[Backpropagation]')
    with st.expander("Gradient y_pred"):
        st.header('Gradient y_pred')
        grad_y_pred = y_pred - Y_matrix  # derivative of categorical cross-entropy
        grad_y_pred_style = pd.DataFrame(grad_y_pred)
        grad_y_pred_style.columns = Y.columns
        grad_y_pred_style.index = x.name
        st.dataframe(grad_y_pred_style.style.apply(highlight, axis=1))

    with st.expander("Gradient W2"):
        st.header('Gradient W2')
        grad_w2 = h.T.dot(grad_y_pred)
        grad_w2_style = pd.DataFrame(grad_w2)
        grad_w2_style.columns = Y.columns
        grad_w2_style.index = w1.columns
        st.dataframe(grad_w2_style.style.apply(highlight, axis=1))



    with st.expander("Gradient h"):
        st.header('Gradient h')
        grad_h = grad_y_pred.dot(w2.T)
        grad_h_style = pd.DataFrame(grad_h)
        grad_h_style.columns = w1.columns
        grad_h_style.index = x.name
        st.dataframe(grad_h_style)




    with st.expander("Gradient Relu"):
        st.header('Gradient Relu')
        grad_relu = reluDerivative(grad_h)
        grad_relu_style = pd.DataFrame(grad_relu)
        grad_relu_style.columns = w1.columns
        grad_relu_style.index = x.name
        st.dataframe(grad_relu_style)


    with st.expander("Gradient W1"):
        st.header('Gradient W1')
        grad_w1 = X_matrix.T.dot(grad_relu)
        grad_w1_style = pd.DataFrame(grad_w1)
        grad_w1_style.columns = w1.columns
        grad_w1_style.index = X.columns
        st.dataframe(grad_w1_style.style.apply(highlight, axis=1))

    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

    st.session_state.w1 = w1
    st.session_state.w2 = w2

    st.subheader('Loss graph')


    st.line_chart(loss_list)
    #st.write(st.session_state.loss_list)

with st.expander("Predict"):
    st.subheader('In prediction we only perform forward propagation')
    with st.form(key='my_form'):
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

        global x_predict
        x_predict = np.array([climate, culture, cuisine, adventure, natural, budget, language, safety])

        predict = st.form_submit_button('Predict')

if predict:

    w1,w2 = st.session_state.w1, st.session_state.w2
    x_predict = (x_predict-mean)/std
    w1_mult_x= x_predict.dot(w1)
    h = np.maximum(0, w1_mult_x)
    y_pred = h.dot(w2)
    y_pred = pd.DataFrame(y_pred)
    y_pred.index = Y.columns
    y_pred = y_pred.sort_values(by=0, ascending=False).index[0]
    st.success(y_pred)
