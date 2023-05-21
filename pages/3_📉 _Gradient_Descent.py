import streamlit as st
import pandas as pd
import graphviz

import altair as alt
import time

import numpy as np

st.set_page_config(page_title="Gradient Descent", page_icon="📉")

st.sidebar.success("Select a demo above.")
st.title("Gradient descent for Neural Nets (simplified example)")



yellow = {
    'style': 'filled',
    'fillcolor': 'yellow'
}

pink = {
    'style': 'filled',
    'fillcolor': 'pink',
    'opacity': '0.5'
}


green = {
    'style': 'filled',
    'fillcolor': 'green',
    'opacity': '0.5',

}

graph_loss = graphviz.Digraph(graph_attr={'rankdir':'LR', 'TBbalance':"max"},format='png')
graph_loss.edge('X', 'Mult')
graph_loss.edge('W1', 'Mult')
graph_loss.edge('Mult', 'Relu')
graph_loss.edge('Relu', 'Prediction')
graph_loss.edge('W2', 'Prediction')
graph_loss.edge('Prediction', 'L2 Loss', label='(y_pred - y_true)**2')
st.graphviz_chart(graph_loss)






with st.form(key="test"):

    col1, col2  = st.columns(2)
    with col1:
        x = st.number_input('X',min_value=0.0, max_value=100.0, value=0.3,key='x',format="%.2f") #3
    with col2:
        y = st.number_input('Y', value=1.0,key='y',format="%.2f") #100

    col3, col4, col5  = st.columns(3)
    with col3:
        w1 = st.number_input('W1', value=0.4,key='w1',format="%.2f") #4
    with col4:
        w2 = st.number_input('W2', value=0.5,key='w2',format="%.2f") #5
    with col5:
        learning_rate = st.number_input('learning rate', value=0.1, step=0.1, format="%.1f", key='learning_rate')
    submit =  st.form_submit_button("Submit")
if submit:
        # forward pass calculation
    w1_mult_x = w1 * x
    h = max(0, w1_mult_x)
    y_pred = w2 * h
    loss = (y - y_pred)**2


    # Backprop calculation
    grad_y_pred = 2*(y_pred -y)
    grad_h = w2*grad_y_pred
    grad_w2 = h*grad_y_pred


    grad_relu = 1 if h > 0 else 0
    grad_q = grad_relu* grad_h

    grad_w1 = x*grad_q
    grad_x = w1*grad_q

    update_w1 =  w1 - learning_rate*grad_w1
    update_w2 = w2 - learning_rate*grad_w2


    graph = graphviz.Digraph(graph_attr={'rankdir':'LR', 'fontsize':"20"})

    # Forward pass
    # X
    graph.edge(f'{x:.2f}', f'{w1_mult_x:.2f} ', label=f'{x:.2f}*{w1:.2f}')

    # W1
    graph.node(f'{w1:.2f}', shape='circle',**yellow)
    graph.edge(f'{w1:.2f}', f'{w1_mult_x:.2f} ', label=f'{w1:.2f}*{x:.2f}')

    # Relu
    graph.edge(f'{w1_mult_x:.2f} ', f'{h:.2f}', label=f'max(0,{w1_mult_x:.2f})')
    graph.edge(f'{h:.2f}', f'{y_pred:.2f}', label=f'{h:.2f}*{w2:.2f}')

    #W2
    graph.node(f'{w2:.2f}', shape='circle',**yellow)

    graph.node(f'{y_pred:.2f}', shape='circle',**green)
    graph.edge(f'{w2:.2f}', f'{y_pred:.2f}', label=f'{w2:.2f}*{h:.2f}', rank='min')

    # Loss
    graph.node(f'{loss:.2f}', shape='circle',**pink)


    graph.edge(f'{y_pred:.2f}', f'{loss:.2f}', label=f'({y_pred:.2f} - {y:.2f})**2')



    ### Backprop
    graph.edge(f'{loss:.2f}', f'{y_pred:.2f}', label=f'2*({y_pred:.2f} - {y:.2f})={grad_y_pred:.2f}', color='red')
    graph.edge(f'{y_pred:.2f}', f'{h:.2f}', label=f'{w2:.2f}*{grad_y_pred:.2f}={grad_h:.2f}', color='red',)
    graph.edge(f'{y_pred:.2f}', f'{w2:.2f}', label=f'{h:.2f}*{grad_y_pred:.2f}={grad_w2:.2f}', color='red',constraint='false')
    graph.edge(f'{h:.2f}', f'{w1_mult_x:.2f} ', label=f'{grad_relu:.2f}*{grad_h:.2f}={grad_q:.2f}', color='red')
    graph.edge(f'{w1_mult_x:.2f} ', f'{x:.2f}', label=f'{grad_q:.2f}*{w1:.2f}={grad_x:.2f}', color='red', )
    graph.edge(f'{w1_mult_x:.2f} ', f'{w1:.2f}', label=f'{grad_q:.2f}*{x:.2f}={grad_w1:.2f}', color='red',  constraint='false')

    st.graphviz_chart(graph)



    st.write(f'Updated W1: {w1:.2f} - ({learning_rate:.2f}*{grad_w1:.2f})= {update_w1:.2f}')
    st.write(f'Updated W2: {w2:.2f} - ({learning_rate:.2f}*{grad_w2:.2f}) = {update_w2:.2f}')


losses = []
for each in range(100):
    w1_mult_x = w1 * x
    h = max(0, w1_mult_x)
    y_pred = w2 * h
    loss = (y - y_pred)**2
    losses.append(loss)


    # Backprop calculation
    grad_y_pred = 2*(y_pred -y)
    grad_h = w2*grad_y_pred
    grad_w2 = h*grad_y_pred


    grad_relu = 1 if h > 0 else 0
    grad_q = grad_relu* grad_h

    grad_w1 = x*grad_q
    grad_x = w1*grad_q

    w1 =  w1 - learning_rate*grad_w1
    w2 = w2 - learning_rate*grad_w2




df = pd.DataFrame({'epoch':np.arange(len(losses)),'value':losses})


def plot_animation(df):
    lines = alt.Chart(df).mark_line().encode(
       x=alt.X('epoch:Q', axis=alt.Axis(title='epoch')),
       y=alt.Y('value:Q',axis=alt.Axis(title='value')),
     ).properties(width=675,height=300)

    return lines


lines = alt.Chart(df).mark_line().encode(
     x=alt.X('1:T',axis=alt.Axis(title='epoch')),
     y=alt.Y('1:Q',axis=alt.Axis(title='value'))
).properties(
    width=675,
    height=300
)

N = df.shape[0] # number of elements in the dataframe
burst = 10       # number of elements (months) to add to the plot
size = burst     # size of the current dataset
line_plot = st.altair_chart(lines)
start_btn = st.button('Start')

if start_btn:
   for i in range(1,N):
      step_df = df.iloc[0:size]
      lines = plot_animation(step_df)
      line_plot = line_plot.altair_chart(lines)
      size = i + burst
      if size >= N:
         size = N - 1
      time.sleep(0.1)
if st.button('Show final results'):
    st.write(f"Final W1: {w1:.2f}, W2: {w2:.2f}")
