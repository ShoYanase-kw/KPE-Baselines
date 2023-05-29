from keybert import KeyBERT

import pathlib
import pandas as pd

from .SuperClass import KeywordExtractor
import MeCab

class KeyBert_(KeywordExtractor):
    def __init__(self, data: pd.DataFrame, thresfold: float = 0.65, topn: int = 5) -> None:
        super().__init__(data, thresfold, topn)

        self.df_scrs_kwds = None
        self.name = "KeyBERT"
        self.model = KeyBERT(f'{pathlib.Path.cwd()}/models/transformers-sentence-bert/sbert_stair')

    def extract_phrases(self, data: pd.Series):
        # MeCabで分かち書き
        tokens = MeCab.Tagger("-Owakati").parse(data["text"])

        words = self.model.extract_keywords(tokens, keyphrase_ngram_range=(1, 1), stop_words=None)
        # スコアで足切り
        self.df_scrs_kwds = pd.DataFrame([w for w in words[:self.topn] if w[1]>self.thresfold], columns=["keyword","score"])
        
        if len(self.df_scrs_kwds) > 0:
            return self.df_scrs_kwds["score"].values.tolist(), self.df_scrs_kwds["keyword"].values.tolist()
        else:
            return [], []