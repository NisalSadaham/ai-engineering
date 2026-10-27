import pandas as pd
import numpy as np 

marks = np.array([
    [78, 65, 82],
    [45, 38, 55],
    [90, 88, 95],
    [32, 48, 40],
    [67, 72, 60],
    [85, 91, 78]
])


df = pd.DataFrame(
    marks,
    columns=["Python","Maths","Data science"]

)

print(df.head())
print(df.shape)
print(df.columns)
print(df[["Python","Maths"]])
print(df.loc[1:3,['Python','Data science']])
experssion = (df["Maths"] < 50 ) | ( df["Python"] > 80)
marks = df.loc[experssion,["Maths","Python"]]
print(df["Maths"].mean())
print(df.mean())

data = {
    "Student": ["A", "B", "C"],
    "Maths": [80, None, 60]
}

df2 = pd.DataFrame(data)
print(df2)
print(df2.isna().sum())
df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce") #to handle data cells with text or other insteead of int to Nan