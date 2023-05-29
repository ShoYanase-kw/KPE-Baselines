import pandas as pd

from .SuperClass import KeywordExtractor
from pke.unsupervised import TopicRank

class TopicRank_(KeywordExtractor):
    def __init__(self, data: pd.DataFrame, thresfold: float = 0.75, topn: int = 5) -> None:
        super().__init__(data, thresfold, topn)

        self.df_scrs_kwds = None
        self.name = "TopicRank"
        self.extractor = TopicRank()

    def extract_phrases(self, data: pd.Series):
        self.extractor.load_document(
            input=data["text"], language="ja", normalization=None
        )
        self.extractor.candidate_selection(pos={"NOUN", "PROPN", "ADJ", "NUM"})
        self.extractor.candidate_weighting()

        self.df_scrs_kwds = pd.DataFrame(self.extractor.get_n_best(n=self.topn), columns=["keyword","score"])

        if len(self.df_scrs_kwds) > 0:
            return self.df_scrs_kwds["score"].values.tolist(), self.df_scrs_kwds["keyword"].values.tolist()
        else:
            return [], []