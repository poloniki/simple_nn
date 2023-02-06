import dash
from dash import html, dash_table
from dash import dcc
import psycopg2
import pandas as pd
from flask import Flask

server = Flask(__name__)
app = dash.Dash(server=server)

conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="docker"
)
with conn:
    cur = conn.cursor()
# query= "SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
# cur.execute(query)
# print(cur.fetchall())
# df = pd.read_sql_query("SELECT * FROM test", conn)
# print(df)

app.layout = html.Div([
    html.H3("Input Form"),
    dcc.Input(id='Name', type='text', value=''),
    dcc.Input(id='Climate', type='number', value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    dcc.Input(id='Culture', type='number', value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    dcc.Input(id='Cuisine', type='number', value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    dcc.Input(id='Adventure activities', type='number', value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    dcc.Input(id='Natural beauty', type='number', value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    dcc.Input(id='Budget', type='number',value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    dcc.Input(id='Language', type='number', value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    dcc.Input(id='Safety', type='number', value='1',min=1,max=10,step=1,placeholder='Enter a value between 1 and 10'),
    html.Button('Submit', id='button'),
    html.Div(id='output')
])

@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('Name', 'value'),
     dash.dependencies.State('Climate', 'value'),
     dash.dependencies.State('Culture', 'value'),
     dash.dependencies.State('Cuisine', 'value'),
     dash.dependencies.State('Adventure activities', 'value'),
     dash.dependencies.State('Natural beauty', 'value'),
     dash.dependencies.State('Budget', 'value'),
     dash.dependencies.State('Language', 'value'),
     dash.dependencies.State('Safety', 'value')])
def update_output(n_clicks,name, input1, input2, input3, input4, input5, input6, input7, input8):
    if input1 and input2  and input3 and input4 and input5 and input6 and input7 and input8 and name :
        with conn:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO ratings (name,climate,culture,cuisine,adventure_activities,natural_beauty,budget,language,safety) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (name, input1, input2, input3, input4, input5, input6, input7, input8))
        df = pd.read_sql_query("SELECT * FROM ratings", conn)
        return html.Div(
        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
            ],
            data=df.to_dict('records'),
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable="multi",
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 20,
        )
            )


@app.callback(
    dash.dependencies.Output('div-out','children'),
    [dash.dependencies.Input('datatable', 'rows'),
     dash.dependencies.Input('datatable', 'selected_row_indices')])
def f(rows,selected_row_indices):
    #either:
    selected_rows=[rows[i] for i in selected_row_indices]
    #or
    selected_rows=pd.DataFrame(rows).iloc[i]
    return 1

if __name__ == '__main__':
    app.run_server(debug=True)
