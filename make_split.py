from common import make_split, CACHE
df = make_split()
df.to_csv("split.csv", index=False)
print(df.split.value_counts().to_dict())
print(df.groupby("split").has_lesion.agg(["sum", "count"]))
