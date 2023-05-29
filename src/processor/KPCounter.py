import pandas as pd
import collections
from tqdm import tqdm

def KPCounter(df: pd.DataFrame, conf, thres: float = 0.75, models:list = ["YAKE", "SGRank", "TextRank", "sCake", "Rake", "MultipartiteRank", "PositionRank", "TopicRank", "KeyBert"]) -> list:
    counts = {}
    for m in models:
        model_info = conf[m]
        model = model_info["class"](df, model_info["thres"], model_info["topn"])
        res = []
        for index, row in tqdm(df.iterrows()):
            scores, words = model.extract_phrases(row)
            res.extend(words)
        c = collections.Counter(res)
        print(model.name, ": ",c)
        counts[m] = c
    return counts
