import streamlit as st
import pandas as pd
import psycopg2
import graphviz

import graphviz as gv
import graphviz as gv

g1 = gv.Graph(graph_attr={'rankdir':'LR', 'TBbalance':"max"},format='png')

g1.attr('node', shape='circle')
g1.node('X')
g1.node('h_prev')
g1.node('U')
g1.node('W')
g1.node('h')
g1.node('V')
g1.node('y')

g1.edge('X', 'U', label='input\ntransform')
g1.edge('h_prev', 'W', label='hidden\ntransform')
g1.edge('U', 'h', label='input\nhidden')
g1.edge('W', 'h', label='prev\nhidden')
g1.edge('h', 'V', label='hidden\noutput')
g1.edge('V', 'y', label='output\ntransform')

g1.render('vanilla_rnn')
st.graphviz_chart(g1)
