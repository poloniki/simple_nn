import streamlit as st
import pandas as pd
import psycopg2
import graphviz

import numpy as np
from numpy.random import randn

def highlight(data):
    is_min = data == data.nsmallest(1).iloc[-1]
    is_max = data == data.nlargest(1).iloc[0]
    styles = [''] * len(data)
    min_index = np.flatnonzero(is_min.to_numpy())
    max_index = np.flatnonzero(is_max.to_numpy())
    for i in min_index:
        styles[i] = 'background-color: red'
    for i in max_index:
        styles[i] = 'background-color: green'
    return styles


graph_loss = graphviz.Digraph(graph_attr={'rankdir':'LR', 'TBbalance':"max"},format='png')
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
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="docker"
)


x = pd.read_sql_query("SELECT * FROM ratings", conn)
y = pd.read_sql_query("SELECT * FROM answers", conn)




expander_X = st.expander("X")
expander_X.dataframe(x)
expander_Y = st.expander("Y (y_true)")
expander_Y.dataframe(y)

X = x.drop('name', axis=1)
Y = y.drop('name', axis=1)
N, D_in, H, D_out = X.shape[0], X.shape[1], 10, Y.shape[1]

X_matrix = X.astype('float').to_numpy()
Y_matrix = Y.astype('float').to_numpy()





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

learning_rate = st.number_input('Learning Rate', value=0.000001,key='learning_rate',format="%.6f")


@st.cache(allow_output_mutation=True)
def get_my_list():
    return []

if initialize:
    if st.session_state.count ==1:
        w1, w2 = randn(D_in, H), randn(H, D_out)
    else:
        w1,w2 = st.session_state.w1, st.session_state.w2

    st.write('Iteration = ', st.session_state.count)


    w1 = pd.DataFrame(w1)
    w1.columns = [f'W1_feature {i}' for i in range(1, w1.shape[1]+1)]
    w1.index = [X.columns]


    w2 = pd.DataFrame(w2)
    w2.columns = Y.columns
    w2.index = [w1.columns]


    expander_W1 = st.expander("W1 with 10 layers")
    expander_W1.dataframe(w1.style.apply(highlight, axis=1))

    expander_W2 = st.expander("W2")
    expander_W2.dataframe(w2)

    st.header("X * W1 | x.dot(w1)")
    w1_mult_x = X_matrix.dot(w1)
    st.dataframe(w1_mult_x)

    st.header("Relu(0,x) | np.maximum(0, w1_mult_x)")
    h = np.maximum(0, w1_mult_x)
    h_style = pd.DataFrame(h)
    st.dataframe(h_style.style.applymap(lambda x: 'background-color: rgba(255,0,0,0.3)' if x == 0 else ''))

    st.header("Relu*W2 | relu.dot(w2)")
    y_pred = h.dot(w2)
    st.dataframe(y_pred)

    st.header('Loss - np.square(y_pred - y_true).sum()')
    loss = np.square(y_pred - Y_matrix).sum()
    st.write(loss)
    loss_list = get_my_list()
    loss_list.append(loss)






    # # Backprop calculation
    st.header('Gradient y_pred')
    grad_y_pred = 2*(y_pred -Y_matrix)
    st.dataframe(grad_y_pred)

    st.header('Gradient h')
    grad_h = grad_y_pred.dot(w2.T)
    st.dataframe(grad_h)

    st.header('Gradient W2')
    grad_w2 = h.T.dot(grad_y_pred)
    st.dataframe(grad_w2)

    st.header('Gradient Relu')
    grad_relu = reluDerivative(grad_h)
    st.dataframe(grad_relu)

    st.header('Gradient W1')
    grad_w1 = X_matrix.T.dot(grad_relu)
    st.dataframe(grad_w1)

    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

    st.session_state.w1 = w1
    st.session_state.w2 = w2

    st.line_chart(loss_list)
    #st.write(st.session_state.loss_list)
