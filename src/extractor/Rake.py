import pandas as pd
import numpy as np
import string

from .SuperClass import KeywordExtractor
from rake_ja import Tokenizer, JapaneseRake

from sklearn.preprocessing import MinMaxScaler


class Rake_(KeywordExtractor):
    def __init__(self, data: pd.DataFrame, thresfold: float = 0.75, topn: int = 5) -> None:
        super().__init__(data, thresfold, topn)

        
        self.df_scrs_kwds = None
        self.name = "Rake"
        self.tokenizer = Tokenizer()
        self.punctuations = string.punctuation + ",.。、"
        self.stopwords = (
            "か な において にとって について する これら から と も が は て で に を は し た の ない よう いる という".split()
            + "により 以外 それほど ある 未だ さ れ および として といった られ この ため こ たち ・ ご覧".split()
        )
        self.rake = JapaneseRake(
            max_length=3,
            punctuations=self.punctuations,
            stopwords=self.stopwords,
        )

    def extract_phrases(self, data: pd.Series):
        tokens = self.tokenizer.tokenize(self._preprocess(data["text"]))

        self.rake.extract_keywords_from_text(tokens)
        self.df_scrs_kwds = pd.DataFrame(self.rake.get_ranked_phrases_with_scores(), columns=["score","keyword"])
        

        if len(self.df_scrs_kwds) > 0:
            mmscaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            mmscaler.fit(np.array(self.df_scrs_kwds["score"].astype('float')).reshape(-1,1))
            
            self.df_scrs_kwds["score"] = mmscaler.transform(np.array(self.df_scrs_kwds["score"]).reshape(-1,1))

            return self.df_scrs_kwds["score"].values.tolist(), self.df_scrs_kwds["keyword"].values.tolist()
        else:
            return [], []