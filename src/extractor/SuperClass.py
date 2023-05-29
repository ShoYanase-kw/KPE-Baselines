import pandas as pd
import neologdn
import re
from tqdm import tqdm


# SuperClass
class KeywordExtractor:
    def __init__(
        self,
        data: pd.DataFrame,
        thresfold: float,
        topn: int,
    ) -> None:

        self.setThresfold(thresfold)
        self.setData(data.fillna(""))
        self.setTopn(topn)
        
    def setData(self, data: pd.DataFrame):
        self.data = data
    
    def getData(self) -> pd.DataFrame:
        return self.data
        
    def setThresfold(self, thresfold: float):
        self.thresfold = thresfold

    def getThresfold(self) -> float:
        return self.thresfold
    
    def setTopn(self, topn: int):
        self.topn = topn

    def getTopn(self) -> int:
        return self.topn

    # 前処理
    def _preprocess(self, x: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )

        x = neologdn.normalize(str(x))

        return x

    def extract_phrases(self, data: pd.Series):
        raise NotImplementedError

    def apply_keywords_extract(self) -> pd.DataFrame:
        tqdm.pandas()
        self.data[["scores", "keywords"]] = self.data.progress_apply(
            self.extract_phrases, axis=1, result_type="expand"
        )

        return self.data