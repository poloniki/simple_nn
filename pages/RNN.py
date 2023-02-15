import streamlit as st
import pandas as pd
import numpy as np
import graphviz as gv
import random
g1 = gv.Graph(graph_attr={'rankdir':'LR', 'TBbalance':"max"})

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







# Define the initial sales data as a list of numbers
data = [10, 12, 15]

# Define the mean and standard deviation of the initial sales data
mean_sales = np.mean(data)
std_sales = np.std(data)

# Define a function to generate new sales data based on the last 3 days
def generate_new_sales(last_sales):
    # Calculate the mean and standard deviation of the last 3 days of sales
    last_mean = np.mean(last_sales)
    last_std = np.std(last_sales)

    # Calculate the new sales value as the mean of the last 3 days plus a random amount
    value = np.round(last_mean+(last_mean*0.12))
    print(last_mean, last_std, last_sales, value)
    return value

# Generate the next 10 days of sales data based on the initial sales data
for i in range(51):
    # Get the last 3 days of sales data
    last_sales = data[-3:]

    # Generate the new sales data based on the last 3 days
    new_sales = generate_new_sales(last_sales)

    # Append the new sales data to the initial sales list
    data.append(new_sales)



column1, column2, column3 = st.columns(3)

with column1:
    st.subheader('Input data')
    num_input_features = 1
    #data = np.array([100, 90, 120, 80, 150, 110, 130, 140, 115, 105, 90, 125, 135, 100, 95])

    df = pd.DataFrame(data, columns=['Ice Cream Sales'])
    st.dataframe(df)

with column2:
    st.subheader('Diff data')
    mean = np.mean(data)
    std = np.std(data)

    # Normalize the array along the columns
    data = (data - mean) / std

    df['diff'] = df['Ice Cream Sales'].diff()
    data = df['diff'].values[1:]
    df = pd.DataFrame(data, columns=['Diff Sales'])
    st.dataframe(df)

with column3:
    st.subheader('Normalized data')
    mean = np.mean(data)
    std = np.std(data)

    # Normalize the array along the columns
    data = (data - mean) / std
    df = pd.DataFrame(data, columns=['Ice Cream Sales'])
    st.dataframe(df)


with st.form(key='my_form'):
    st.write('Number of Time Stamps = Sequence Length = Number of RNN cells. How many days of data do we need to pedict sales?')
    num_time_steps = st.number_input(label='Number of time steps', min_value=2, max_value=5, step=1,value=3)

    st.write('Batch size = Number of sequences in a batch. How many samples do we pass at once at a time?')
    batch_size = st.number_input(label='Batch size', min_value=1, max_value=20, step=1, value=10)

    st.write('Number of hidden units = Number of neurons in a cell. How many new features that pass through time we want to create? Example: keras.layers.RNN(10) - means 10 neurons = 10 new features')
    num_hidden_units = st.number_input(label='Neurons', min_value=1, max_value=10, step=1, value=4)

    st.write('Learning rate')
    learning_rate = st.number_input(label='Learning rate', min_value=0.0001, max_value=0.1, step=0.0001, value=0.01)


    submit_button = st.form_submit_button(label='Submit')

def create_df(array, columns, index, name):
    df = pd.DataFrame(array, columns=columns)
    df.index = index
    st.write(f'{name}. Shape:', array.shape)
    return st.dataframe(df)

st.header('Initialize the weight matrices and bias vectors for the RNN layer')
with st.expander('Weights and Bias'):
    W_xh = np.random.randn(num_input_features, num_hidden_units)
    create_df(W_xh, [f'W_xh_{i}' for i in range(num_hidden_units)], [f'{df.columns[0]}' for i in range(num_input_features)], 'W_xh')

    W_hh = np.random.randn(num_hidden_units, num_hidden_units)
    create_df(W_hh, [f'W_hh_{i}' for i in range(num_hidden_units)], [f'feature_{i}' for i in range(num_hidden_units)], 'W_hh')

    b_h = np.zeros((1, num_hidden_units))
    #create_df(b_h, [f'b_h_{i}' for i in range(num_hidden_units)], [f'1' for i in range(1)], 'b_h')

    W_hy = np.random.randn(num_hidden_units, num_input_features)
    create_df(W_hy, [f'W_hy_{i}' for i in range(num_input_features)], [f'feature_{i}' for i in range(num_hidden_units)], 'W_hy')

    b_y = np.zeros((1, num_input_features))
    #create_df(b_y, [f'b_y_{i}' for i in range(num_input_features)], [f'1' for i in range(1)], 'b_y')
st.header('Initialize the hidden state (this is the initial state before processing any input)')
with st.expander('Hidden State'):
    h_t = np.zeros((batch_size, num_hidden_units))
    create_df(h_t, [f'feature_{i}' for i in range(num_hidden_units)], [f'batch_sample_{i}' for i in range(batch_size)], 'h_t')


if 'count' not in st.session_state:
	st.session_state.count = 0

def increment_counter():
	st.session_state.count += 1

col1, col2, col3  = st.columns(3)

with col1:
    initialize =  st.button('Epoch iteration',on_click=increment_counter)
# Loop through the input sequence and update the hidden state at each time step
if initialize:
    st.write('Epoch = ', st.session_state.count)
    if st.session_state.count ==1:
        pass
    else:
        W_xh, W_hh, b_h, W_hy, b_y = st.session_state.W_xh, st.session_state.W_hh, st.session_state.b_h, st.session_state.W_hy, st.session_state.b_y
    for i in range(0, len(data) - num_time_steps, batch_size):
        st.header(f'------------ Batch {int(i/batch_size)} ------------')
        # Extract the input sequence for the current batch
        x_batch = np.zeros((batch_size, num_time_steps, num_input_features))

        for j in range(batch_size):
            x_batch[j,:,:] = data[i+j:i+j+num_time_steps].reshape((num_time_steps, num_input_features))

        # Compute the new hidden states for the current batch using the current inputs and previous hidden states
        h_t = np.zeros((batch_size, num_hidden_units))
        st.header(f':green[----------Forward pass----------]')
        column1, column2 = st.columns(2)
        for t in range(num_time_steps):
            st.subheader(f':green[RNN cell {t}]')
            with st.expander('Cell'):
                x_t = x_batch[:,t,:]
                create_df(x_t, [f'{df.columns[i]}' for i in range(num_input_features)], [f'batch_sample_{i}' for i in range(batch_size)], 'x_t')
                h_t = np.tanh(np.dot(x_t, W_xh) + np.dot(h_t, W_hh) + b_h)
                create_df(h_t, [f'feature_{i}' for i in range(num_hidden_units)], [f'batch_sample_{i}' for i in range(batch_size)], 'h_t')


        # Compute the outputs for the current batch at the last time step
        st.subheader(f':pink[Prediction]')
        with st.expander('Prediction'):
            column_y1, column_y2 = st.columns(2)
            with column_y1:
                st.write('Normalized')
                y_pred = np.dot(h_t, W_hy) + b_y
                create_df(y_pred, [f'y_{i}' for i in range(num_input_features)], [f'batch_sample_{i}' for i in range(batch_size)], 'y_pred')
                # Compute the target values for the current batch (in this case, we want to predict the next day's sales based on the previous 3 days of sales)
                y_true = data[i+num_time_steps:i+num_time_steps+batch_size].reshape((batch_size, num_input_features))
                create_df(y_true, [f'y_{i}' for i in range(num_input_features)], [f'batch_sample_{i}' for i in range(batch_size)], 'y_true')

            with column_y2:
                st.write('Unnormalized')
                unnormalized_y_pred = y_pred * std + mean
                create_df(unnormalized_y_pred, [f'y_{i}' for i in range(num_input_features)], [f'batch_sample_{i}' for i in range(batch_size)], 'unnormalized_y')
                unnormalized_y_true = y_true * std + mean
                create_df(unnormalized_y_true, [f'y_{i}' for i in range(num_input_features)], [f'batch_sample_{i}' for i in range(batch_size)], 'unnormalized_y_true')

        # Compute the error (mean squared error) for the current batch
        error = 0.5 * np.mean((y_pred - y_true) ** 2)
        unnormalized_error = 0.5 * np.mean((unnormalized_y_pred - unnormalized_y_true) ** 2)
        error_column1, error_column2 = st.columns(2)
        with error_column1:
            st.write('Error')
            st.warning(error)
        with error_column2:
            st.write('Unnormalized Error')
            st.warning(unnormalized_error)

        # Compute the gradients of the output layer for the current batch (using the chain rule)
        grad_y = (y_pred - y_true) / batch_size
        grad_W_hy = np.dot(h_t.T, grad_y)
        grad_b_y = np.sum(grad_y, axis=0, keepdims=True)

        # Initialize the gradients of the hidden state for the current batch (this will be used as the initial gradients for backpropagation)
        grad_h = np.zeros((batch_size, num_hidden_units))
        st.header(f':red[----------Backward pass----------]')
        # Loop backward through the time steps and compute the gradients for each time step for the current batch
        for t in reversed(range(num_time_steps)):
            st.subheader(f':red[Gradient of RNN cell {t}]')
            with st.expander('Cell'):
                x_t = x_batch[:,t,:]
                h_t = np.tanh(np.dot(x_t, W_xh) + np.dot(h_t, W_hh) + b_h)

                # Compute the gradients for the output of the RNN layer (using the chain rule)
                grad_output = grad_h + np.dot(grad_y, W_hy.T)
                create_df(grad_output, [f'grad_output_{i}' for i in range(num_hidden_units)], [f'batch_sample_{i}' for i in range(batch_size)], 'grad_output')
                grad_z = grad_output * (1 - h_t ** 2)
                create_df(grad_z, [f'grad_z_{i}' for i in range(num_hidden_units)], [f'batch_sample_{i}' for i in range(batch_size)], 'grad_z')

                # Compute the gradients for the parameters of the RNN layer (using the chain rule)
                grad_W_xh = np.dot(x_t.T, grad_z)
                create_df(grad_W_xh, [f'grad_W_xh_{i}' for i in range(num_hidden_units)], [f'x_{i}' for i in range(num_input_features)], 'grad_W_xh')
                grad_W_hh = np.dot(h_t.T, grad_z)
                create_df(grad_W_hh, [f'grad_W_hh_{i}' for i in range(num_hidden_units)], [f'h_{i}' for i in range(num_hidden_units)], 'grad_W_hh')
                grad_b_h = np.sum(grad_z, axis=0, keepdims=True)

                # Update the gradients for the next time step (using the chain rule)
                grad_h = np.dot(grad_z, W_hh.T)
                create_df(grad_h, [f'grad_h_{i}' for i in range(num_hidden_units)], [f'batch_sample_{i}' for i in range(batch_size)], 'grad_h')

            # Accumulate the gradients for the current batch
            if t == num_time_steps - 1:
                total_grad_W_xh = grad_W_xh
                total_grad_W_hh = grad_W_hh
                total_grad_b_h = grad_b_h
                total_grad_W_hy = grad_W_hy
                total_grad_b_y = grad_b_y
            else:
                total_grad_W_xh += grad_W_xh
                total_grad_W_hh += grad_W_hh
                total_grad_b_h += grad_b_h
                total_grad_W_hy += grad_W_hy
                total_grad_b_y += grad_b_y

        # Update the parameters of the RNN layer for the current batch using the computed gradients and the learning rate
        W_xh -= learning_rate * total_grad_W_xh
        W_hh -= learning_rate * total_grad_W_hh
        b_h -= learning_rate * total_grad_b_h
        W_hy -= learning_rate * total_grad_W_hy
        b_y -= learning_rate * total_grad_b_y

        st.session_state.W_xh = W_xh
        st.session_state.W_hh = W_hh
        st.session_state.b_h = b_h
        st.session_state.W_hy = W_hy
        st.session_state.b_y = b_y
