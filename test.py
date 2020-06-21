import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

corr_mat_file_name = "data/all/correlation.h5"
groups = "data/all/groups.pkl"

with open(groups, "rb") as groups_file:
    groups = pickle.load(groups_file)

corr_mat = pd.read_hdf(corr_mat_file_name)

shuff = corr_mat.sample(frac=1)
shuff.index = shuff.columns

# corr_mat = shuff

std = corr_mat.values.std(ddof=1)

means = []

for group in groups:
    group_corr = corr_mat.loc[group, group]
    mean_corr = group_corr.mean().mean()
    means.append(mean_corr/std)

print(np.mean(means))

plt.hist(means, bins=20)
plt.show()