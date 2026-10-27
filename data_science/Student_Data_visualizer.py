import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Student": ["Alex", "Ben", "Charlie", "Diana", "Ethan",
                "Fiona", "George", "Hannah", "Ivan", "Julia"],
    "Python": [78, 45, 90, 32, 67, 85, 55, 72, 88, 61],
    "Maths": [65, 38, 88, 48, 72, 91, 55, 47, 82, 76],
    "Data Science": [82, 55, 95, 40, 60, 78, 68, 52, 89, 73]
}

df = pd.DataFrame(data)

print(df.head())
print(df.describe())

#1
subject_averages = df[["Python","Maths","Data Science"]].mean()
print(subject_averages)

#2
student_average = df[["Python","Maths","Data Science"]].mean(axis=1)
print(student_average)
index = student_average.idxmax()
print(f"Highest Average Student:{df.loc[index,"Student"]}")
print(f"Highest Average:{student_average.max()}")
#3
plt.bar(df["Student"],student_average)
plt.title("Average across three subjects")
plt.xlabel("Student Name")
plt.ylabel("Average")
plt.xticks(rotation=45)
plt.show()
#4
plt.scatter(df['Python'],df['Maths'])
plt.title("Maths vs Python marks")
plt.xlabel("Python")
plt.ylabel("Maths")
plt.show()
#5
plt.hist(student_average,bins=5)
plt.title("distribution of students' average marks")
plt.xlabel("Average marks")
plt.ylabel("No of students")
plt.show()


