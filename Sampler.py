import numpy as np
from aif360.algorithms import Transformer
from aif360.datasets import BinaryLabelDataset
from pandas.core.frame import DataFrame


def balance_set(w_exp, w_obs, df, tot_df, round_level=None, debug=False):
    disp = round(w_exp / w_obs, round_level) if round_level else w_exp / w_obs
    disparity = [disp]
    while disp != 1:
        if w_exp / w_obs > 1:
            df = df.append(df.sample())
        elif w_exp / w_obs < 1:
            df = df.drop(df.sample().index, axis=0)
        w_obs = len(df) / len(tot_df)
        disp = round(
            w_exp / w_obs, round_level) if round_level else w_exp / w_obs
        disparity.append(disp)
        if debug:
            print(w_exp / w_obs)
    return df, disparity


def sample(d: DataFrame, s_vars: list, label: str, round_level: float, debug: bool = False, i: int = 0, G: list = [], cond: bool = True):
    d = d.copy()
    n = len(s_vars)
    disparities = []
    if i == n:
        for l in np.unique(d[label]):
            g = d[(cond) & (d[label] == l)]
            w_exp = (len(d[cond])/len(d)) * (len(d[d[label] == l])/len(d))
            w_obs = len(g)/len(d)
            g_new, disp = balance_set(w_exp, w_obs, g, d, round_level, debug)
            disparities.append(disp)
            G.append(g_new)
        return G
    else:
        s = s_vars[i]
        i = i+1
        G1 = sample(d, s_vars, label, round_level, debug, i,
                    G.copy(), cond=cond & (d[s] == 0))
        G2 = sample(d, s_vars, label, round_level, debug, i,
                    G.copy(), cond=cond & (d[s] == 1))
        G += G1
        G += G2
        if len(G) == 2**(n+1):
            return DataFrame(G.pop().append([g for g in G]).sample(frac=1))
        else:
            return G


class Sampler(Transformer):
    def __init__(self, round_level=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.round_level = round_level
        self.debug = debug

    def predict(self, dataset):
        return dataset

    def transform(self, dataset):
        return dataset

    def fit_transform(self, dataset: BinaryLabelDataset):

        df_new = sample(dataset.convert_to_dataframe()[0], dataset.protected_attribute_names,
                        dataset.label_names[0], self.round_level, self.debug, 0, [], True)
        return BinaryLabelDataset(df=df_new, protected_attribute_names=dataset.protected_attribute_names, label_names=dataset.label_names)
