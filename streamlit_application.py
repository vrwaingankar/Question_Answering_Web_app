import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image
from haystack.nodes import FARMReader

new_reader = FARMReader(model_name_or_path="my_model")

def welcome():
    return "Welcome All"

def predict_answer(Question,context):
    prediction=new_reader.predict_on_texts(Question,[context])
    print(prediction)
    return prediction

def main():
    st.title("Question Answering")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Questioning Answering ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Context = st.text_input("Context","Type Here")
    Question = st.text_input("Question","Type Here")
    result=""
    if st.button("Answer"):
        result=predict_answer(Context,Question)
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()
