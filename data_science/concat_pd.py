import pandas as pd

class_a = pd.DataFrame({
    "Student": ["Alex", "Charlie"],
    "Marks": [80, 90]
})

class_b = pd.DataFrame({
    "Student": ["Ben", "Diana"],
    "Marks": [65, 55]
})

all_students = pd.concat([class_a,class_b],ignore_index=True)
print(all_students)

student_names = pd.DataFrame({
    "Student": ["Alex", "Ben"]
})

student_marks = pd.DataFrame({
    "Marks": [80, 65]
})

student_records = pd.concat([student_names,student_marks],axis=1)
print(student_records)

#merge
students = pd.DataFrame({
    "StudentID": [101, 102, 103],
    "Student": ["Alex", "Ben", "Charlie"]
})

marks = pd.DataFrame({
    "StudentID": [101, 102],
    "Marks": [80, 65]
})

student_results = pd.merge(students,marks,on="StudentID")
print(student_results)

all_student_results = pd.merge(students,marks,on="StudentID",how= "left")
print(all_student_results)

all_records = pd.merge(students,marks,on="StudentID",how= "outer")
print(all_records)