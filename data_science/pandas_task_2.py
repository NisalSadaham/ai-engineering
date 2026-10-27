import pandas as pd
data = {
    "Student": ["Alex", "Ben", "Charlie", "Diana", "Ethan", "Fiona"],
    "Class": ["A", "B", "A", "B", "A", "B"],
    "Subject": ["Python", "Python", "Maths", "Maths", "Python", "Maths"],
    "Marks": [80, 65, 90, 55, 75, 85]
}

df = pd.DataFrame(data)

average_marks_each_class = df.groupby('Class')["Marks"].mean()
print(average_marks_each_class)
average_class_subject = df.groupby(["Class","Subject"])["Marks"].mean()
sorted = average_class_subject.sort_values(ascending=False)
print(sorted.head(2).reset_index())
