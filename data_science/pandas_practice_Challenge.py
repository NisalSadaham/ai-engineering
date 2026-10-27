import pandas as pd

students = pd.DataFrame({
    "StudentID": [101, 102, 103, 104, 105],
    "Student": ["Alex", "Ben", "Charlie", "Diana", "Ethan"],
    "Class": ["A", "A", "B", "B", "A"]
})

marks = pd.DataFrame({
    "StudentID": [103, 101, 105, 106],
    "Marks": [90, 80, 72, 88]
})

all_records = pd.merge(students,marks,on="StudentID",how="left")
print(all_records)
sorted_records = all_records.sort_values('Marks',ascending=False)
print(sorted_records.head(3))
