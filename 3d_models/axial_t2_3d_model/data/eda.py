import pandas as pd

df = pd.read_csv("train_all.csv")
new_df = df[["study_id", "left_subarticular_stenosis_l1_l2", "right_subarticular_stenosis_l1_l2", "fold"]]
for index, row in new_df.iterrows():
    if row['left_subarticular_stenosis_l1_l2'] != row['right_subarticular_stenosis_l1_l2']:
        print(row["study_id"], row['left_subarticular_stenosis_l1_l2'], row['right_subarticular_stenosis_l1_l2'], row['fold'])
print(new_df.groupby(["fold"]).value_counts())