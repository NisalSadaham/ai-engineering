import pandas as pd

data = {
    "Student": ["Alex", "Ben", "Charlie", "Diana", "Ethan", "Fiona"],
    "Class": ["A", "B", "A", "B", "A", "B"],
    "Subject": ["Python", "Python", "Maths", "Maths", "Python", "Maths"],
    "Marks": [80, 65, 90, 55, 75, 85]
}

df = pd.DataFrame(data)

print(df)
average = df.groupby("Class")["Marks"].mean()
print(average)

total_marks = df.groupby("Class")["Marks"].sum()
print(total_marks)

no_students = df.groupby("Class")["Student"].count()
print(no_students)

class_report = df.groupby("Class")["Marks"].agg(["mean","sum","count"])
print(class_report)

subject_totals = df.groupby(["Class","Subject"])["Marks"].mean().reset_index()
print(subject_totals)