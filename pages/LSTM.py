import streamlit as st
import pandas as pd
import psycopg2
import graphviz

import graphviz as gv

dot = gv.Digraph(comment='LSTM Computations')

dot.node('x', 'Input\n(x)')
dot.node('h_prev', 'Previous Hidden State\n(h_{t-1})')
dot.node('concat', 'Concatenation\n[h_{t-1}, x]')
dot.node('Wf', 'Weight Matrix\nWf')
dot.node('bf', 'Bias Vector\nbf')
dot.node('ft', 'Forget Gate\nft = sigmoid(Wf * concat + bf)')
dot.node('Wi', 'Weight Matrix\nWi')
dot.node('bi', 'Bias Vector\nbi')
dot.node('it', 'Input Gate\nit = sigmoid(Wi * concat + bi)')
dot.node('Wc', 'Weight Matrix\nWc')
dot.node('bc', 'Bias Vector\nbc')
dot.node('ct_', 'Candidate Memory\nct_ = tanh(Wc * concat + bc)')
dot.node('ct', 'Cell State\nct = ft * c_{t-1} + it * ct_')
dot.node('Wo', 'Weight Matrix\nWo')
dot.node('bo', 'Bias Vector\nbo')
dot.node('ot', 'Output Gate\not = sigmoid(Wo * concat + bo)')
dot.node('ht', 'Hidden State\nht = ot * tanh(ct)')

dot.edge('x', 'concat', 'Append\n[h_{t-1}, x]')
dot.edge('h_prev', 'concat', 'Append\n[h_{t-1}, x]')
dot.edge('concat', 'ft', 'Compute ft\nft = sigmoid(Wf * concat + bf)')
dot.edge('Wf', 'ft', 'Compute ft\nft = sigmoid(Wf * concat + bf)')
dot.edge('bf', 'ft', 'Compute ft\nft = sigmoid(Wf * concat + bf)')
dot.edge('concat', 'it', 'Compute it\nit = sigmoid(Wi * concat + bi)')
dot.edge('Wi', 'it', 'Compute it\nit = sigmoid(Wi * concat + bi)')
dot.edge('bi', 'it', 'Compute it\nit = sigmoid(Wi * concat + bi)')
dot.edge('concat', 'ct_', 'Compute ct_\nct_ = tanh(Wc * concat + bc)')
dot.edge('Wc', 'ct_', 'Compute ct_\nct_ = tanh(Wc * concat + bc)')
dot.edge('bc', 'ct_', 'Compute ct_\nct_ = tanh(Wc * concat + bc)')

dot.edge('ft', 'ct', 'Compute ct\nct = ft * c_{t-1} + it * ct_')
dot.edge('h_prev', 'ct', 'Compute ct\nct = ft * c_{t-1} + it * ct_')
dot.edge('it', 'ct', 'Compute ct\nct = ft * c_{t-1} + it * ct_')
dot.edge('ct_', 'ct', 'Compute ct\nct = ft * c_{t-1} + it * ct_')
dot.edge('concat', 'ot', 'Compute ot\not = sigmoid(Wo * concat + bo)')
dot.edge('Wo', 'ot', 'Compute ot\not = sigmoid(Wo * concat + bo)')
dot.edge('bo', 'ot', 'Compute ot\not = sigmoid(Wo * concat + bo)')
dot.edge('ct', 'ht', 'Compute ht\nht = ot * tanh(ct)')
dot.edge('ot', 'ht', 'Compute ht\nht = ot * tanh(ct)')

st.graphviz_chart(dot)
