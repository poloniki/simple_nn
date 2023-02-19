import streamlit as st


with st.expander('What is a loss function?'):
    st.write('Way to quantify how good a model is at predicting the target variable')
with st.expander('What is SVM Loss'):
    st.write('We take sum of all incorrect predictions and subtract the score for the correct prediction. If this sum is less than 1, we set it to 0. This is the hinge loss.')
with st.expander('What is Softmax Loss'):
    st.write('''Take all scores. Exponentiate them to make them all positive.
             Then we devide by the sum of all scores. This gives probabilities between 0 an 1.
             Loss is the negative log of the probability of the correct class.
             We want our predictions to be 1 for the correct class and 0 for the incorrect classes.
             if we maximmiz log of probability. Loss function measure badness so we need to add -1.
             Mathematicaly it is easier to maximize log, than maximaze row probability.''')

st.write('SVM only cares about going above threshold. Softmax will try to push score for correct class to infinity. In practice they tend to perform similary.')
st.write('Gradient is a vector of partial derivatives. It points to the direction of greates increase. It tells us how much each parameter affects the final result.')

with st.expander('Why do we need regularization?'):
    st.write('Occams Razor: "Among competing hypotheses, the simplest is the best". To prevent overfitting. Actually single problem can have multiple solutions. If you found perfect weights for W which results in loss 0, that does not automatically mean you have the best mode. Your W*2 would still be the solution.')
with st.expander('What is L1 and L2 - what is the difference'):
    st.write('''L2 is most common. Euclidian norm (). Measures complexity of model. Spreads influence between weights.

             Sometimes L1 - incourage sparsity. Elastic - combination. Dropout. Batch normalization.
             L1 has different notion. model is less complex if it has more 0 in the vector. ''')



st.write('Addition - distrubutes. Max - routes. Multiplication - switches.')


st.write('''Sigmoid - squashes number from 0 to 1.
         Problems
         1. Saturated gradient can kill neuron.
         Because of chain rule it can kill flow down the chain. d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
         2. Not zero centered. Our Local gradient will all postive or all negative.

         Tanh - squashes between -1 and 1 - so no second problem. But still problem with saturation d/dx tanh(x) = 1 - tanh(x)^2

         Relu - Does not saturate in + region. Simple. 6x times faster convergence.
         Not zero-centered. Saturared with negative. Possible to have dead relus (if lr is too high )

         Leaky Relu - does not saturate
         ELU
         Maxout
         ''')

st.write(' Why normalize - every feature should have same influence on the start. Optimizing gradient routes.')
st.write(' Why not set all weights to 0 - then they all be updated the same way')
