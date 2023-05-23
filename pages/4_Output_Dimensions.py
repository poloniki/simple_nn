import streamlit as st

def calculate_output_size(W, F, P, S):
    return ((W - F + 2*P) // S) + 1

st.title('Convolution Output Size Calculator')

# Write the formula and some explanation
st.write("""
The formula to calculate the output size of a convolutional layer is:

`(W - F + 2P) / S + 1`

Where:
- W is the input size
- F is the filter size
- P is the padding
- S is the stride
""")

# Use form to group the input fields
with st.form(key='convolution_form'):
    col1,col2,col3,col4 = st.columns(4)

    with col1:

        W = st.number_input('Input size (Height/Width)', min_value=0, value=255, max_value=10000)
    with col2:
        F = st.number_input('Filter size (Height/Width)', min_value=0, value=3, max_value=10000)
    with col3:
        P = st.number_input('Padding (P)', min_value=0, value=0, max_value=10000)
    with col4:
        S = st.number_input('Stride (S)', min_value=1, value=2, max_value=10000)

    submit_button = st.form_submit_button(label='Calculate')

    if submit_button:
        output_size = calculate_output_size(W, F, P, S)
        st.write(f'The output shape is {output_size}x{output_size}')


# Keras example
st.subheader("Here's an example of a Conv2D layer in Keras")

st.code("""
## Keras Conv2D Layers Example


```python
from keras.models import Sequential
from keras.layers import Conv2D

# Create a model
model = Sequential()

# Add a convolutional layer with 32 filters of size 3x3, stride 1, no padding
# Assuming an input size of 64x64, the output size will be 62x62
model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', input_shape=(64, 64, 3)))

# Add another convolutional layer with 64 filters of size 3x3, stride 1, no padding
# The output size will be 60x60
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))
""")
