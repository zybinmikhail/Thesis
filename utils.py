import pandas as pd


def get_tr_data(healthy):
    """
    Read transcriptomic data and transform it to a convenient form.
    """
    healthy_str = "" if healthy else "_SZ"
    peaks = pd.read_csv(
        f"../data/ours_maria_version/new{healthy_str}_ourRNA-35reg-prot_genes-peaks.txt", sep="\t")
    meta = pd.read_csv(
        f"../data/ours_maria_version/new{healthy_str}_ourRNA-35reg-prot_genes-data.txt", sep="\t", header=None)
    data = pd.merge(meta[[1, 5, 7]], peaks.T.reset_index(), left_on=1, right_on="index")
    data = data.rename(columns={1: "batch", 5: "human", 7: "region"}).drop(columns=["index"])
    data = data.set_index("human")
    data = data.sort_values(["human", "region"])
    data.region = data.region.str.replace(u'\xa0', ' ')
    return data


def get_lipids_data(healthy=True):
    """
    Read lipids data and transform it to a convenient form.
    """
    healthy_str = "_H" if healthy else "_SZ"
    lipid = pd.read_csv(f"../data/rtmz{healthy_str}_pos_std_weight_norm_TL_COMBINED.csv", index_col=0).reset_index(names="Sample")
    meta = pd.read_csv("../data/meta_pos_COMBINED.csv", index_col=0)
    meta.Sample = meta.Sample.str.rstrip(".mzXML")
    lipid.Sample = lipid.Sample.str.rstrip(".mzXML")
    lipid = pd.merge(meta[["Sample", "Region_detailed", "Brain_abbr"]], lipid, on="Sample")
    lipid = lipid.rename(columns={"Sample": "batch", "Region_detailed": "region", "Brain_abbr": "human"})
    return lipid
