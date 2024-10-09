import pandas as pd
import numpy as np

soft_df = pd.read_csv("data/oof_exp_26.csv")
df = pd.read_csv("data/train_all.csv")
col_names = df.columns.tolist()[1:-1]

for index, row in df.iterrows():
    study_id = row["study_id"]
    label = row[col_names].values
    new_label = []
    for i in range(label.shape[0]):
        if type(label[i]) == str:
            new_label.append(np.array([float(num) for num in label[i].split(' ')]))
        else:
            new_label.append(label[i])
    label = np.array(new_label)
    print(label.shape)
