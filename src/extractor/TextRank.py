import pandas as pd

from .SuperClass import KeywordExtractor
import textacy
from textacy.extract.keyterms import textrank

class TextRank_(KeywordExtractor):
    def __init__(self, data: pd.DataFrame, thresfold: float = 0.75, topn: int = 5) -> None:
        super().__init__(data, thresfold, topn)

        self.df_scrs_kwds = None
        self.name = "TxtRank"
        self.ja = textacy.load_spacy_lang("ja_core_news_sm")

    def extract_phrases(self, data: pd.Series):
        doc = textacy.make_spacy_doc(self._preprocess(data["text"]), lang=self.ja)
        keywords_with_score = [
            (kps, score) for kps, score in textrank(doc, normalize="lemma", topn=self.topn) if score<1-self.thresfold
        ]

        self.df_scrs_kwds = pd.DataFrame(keywords_with_score, columns=["keyword", "score"])

        if len(self.df_scrs_kwds) > 0:
            return self.df_scrs_kwds["score"].values.tolist(), self.df_scrs_kwds["keyword"].values.tolist()
        else:
            return [], []