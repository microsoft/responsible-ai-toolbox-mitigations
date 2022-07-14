import os

import pickle

from raimitigations.dataprocessing import SeqFeatSelection


def test_pickle(df_full, label_col_name):
    feat_sel = SeqFeatSelection(n_jobs=4)
    feat_sel.fit(df=df_full, label_col=label_col_name)
    new_df = feat_sel.transform(df_full)

    file_writer = open("seq_feat.obj", "wb")
    pickle.dump(feat_sel, file_writer)
    file_writer.close()

    file_reader = open("seq_feat.obj", "rb")
    feat_sel_loaded = pickle.load(file_reader)
    file_reader.close()

    new_df_loaded = feat_sel_loaded.transform(df_full)

    os.remove("seq_feat.obj")

    def lists_equal(l1, l2):
        if len(l1) == len(l2) and len(l1) == sum([1 for i, j in zip(l1, l2) if i == j]):
            return True
        else:
            return False

    assert lists_equal(
        new_df.columns.to_list(), new_df_loaded.columns.to_list()
    ), "ERROR: feature selection object has different set of selected features after loaded."
