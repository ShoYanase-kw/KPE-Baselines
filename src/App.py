from processor.KPCounter import KPCounter
from extractor import *

import pandas as pd
import numpy as np

import streamlit as st

import pathlib

import altair as alt

model_config = {
    "YAKE": {
            "class": YAKE.YAKE_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "SGRank": {
            "class": SGRank.SGRank_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "TextRank": {
            "class": TextRank.TextRank_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "sCake": {
            "class": sCake.sCAKE_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "Rake": {
            "class": Rake.Rake_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "MultipartiteRank": {
            "class": MultipartiteRank.MultipartiteRank_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "PositionRank": {
            "class": PositionRank.PositionRank_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "TopicRank": {
            "class": TopicRank.TopicRank_,
            "thres": 0.75,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        }, 
    "KeyBert": {
            "class": KeyBert.KeyBert_,
            "thres": 0.65,
            "topn": 5,
            "n-gram": 1,
            "active":False,
        },
}

def renderSideBar():
    with st.sidebar:
        st.title('Models')
        for name in list(model_config.keys()):
          model_config[name]["active"] = st.checkbox(name, key=name+"use")
          if model_config[name]["active"]:
            with  st.expander("config"):
                threshold = st.slider(
                    "Threshold",
                    min_value=0.01,
                    max_value=1.00,
                    value=model_config[name]["thres"],
                    key=name+"slider",
                )
                model_config[name]["thres"] = threshold
                
                topn = st.number_input('top-N phrases', value=int(model_config[name]["topn"]), format="%i", key=name+"topn", min_value=1)
                model_config[name]["topn"] = int(topn)
                
                ngrams = st.number_input('N-grams window size', value=int(model_config[name]["n-gram"]), format="%i", key=name+"n-gram", min_value=1)
                model_config[name]["topn"] = int(ngrams)

        models = [n for n in list(model_config.keys()) if model_config[n]["active"]]
        return models

def DLButton(df):
    @st.cache
    def convert_df(df_):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df_.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='KeyPhrase.csv',
        mime='text/csv',
    )
        

def main():
    st.title('KeyPhrase Extraction')

    models = renderSideBar()
    
    uploaded_file = st.file_uploader("Choose a file", type=['txt','csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        object_col_list = df.select_dtypes('object').columns.tolist()
        
        st.write(df.head())
        
        if st.button("Submit"):
            if len(object_col_list)==0:
                st.warning('文字列が見つかりません')
            else:
                df["text"] = df[object_col_list[0]].str.cat([df[n] for n in object_col_list], sep="。")
                
                models = [n for n in list(model_config.keys()) if model_config[n]["active"]]
                kp_count = KPCounter(df=df, conf=model_config, models=models)
                # print([["a"]*len(kp_count)])
                # print([np.array([[m]*len(kp_count[m].keys()), list(kp_count[m].keys()), list(kp_count[m].values())]).T for m in list(kp_count.keys())])
                df_count = pd.concat(
                    [
                        pd.DataFrame(
                            np.array([[m]*len(kp_count[m].keys()), list(kp_count[m].keys()), list(kp_count[m].values())]).T,
                            columns=["model","word","count"]
                        ) for m in list(kp_count.keys())
                    ]
                )
                
                df_count['count'] = df_count['count'].astype('int')
                df_count.sort_values('count')
                    
                # st.write(df_count)
                chart = alt.Chart(df_count).mark_bar().encode(
                    alt.X('count'),
                    alt.Y('word', sort=alt.EncodingSortField(field="count", op="sum", order='descending')),
                    color='model'
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                DLButton(df_count)
        
if __name__ == "__main__":
    # df_sample = pd.read_csv(f'{pathlib.Path.cwd()}/src/sample/chABSA_sentences.csv', header=None, names=('text',)).head()
    # print(df_sample)
    # KPCounter(df=df_sample, conf=model_config)
    main()