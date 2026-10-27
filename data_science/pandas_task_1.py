import pandas as pd
import numpy as np
#task 2
df = pd.read_csv('students.csv')
first_five_rows = df.head()
shape = df.shape
column_names = df.columns
print(f"First five rows:{first_five_rows}")
print(f"Shape:{shape}")
print(f"column names:{column_names}")
print(df.info())
print(f"DF describe:{df.describe()}")

#task 3
count_missing_values = df.isna().sum()
print(f"Missing values:{count_missing_values}")
print(f"non numeric columns:{df.select_dtypes(exclude=['number']).columns}")
#conversion to numerics 

df['Maths'] = pd.to_numeric(df['Maths'],errors='coerce')
df['Python'] = pd.to_numeric(df['Python'],errors='coerce')
df['Data Science'] = pd.to_numeric(df['Data Science'],errors='coerce')
count_missing_values = df.isna().sum()
print(f"Missing values:{count_missing_values}")
cleaned_df = df.copy()
print(f"cleaned dataframe {cleaned_df}")
#task 4
average = cleaned_df[["Maths","Python","Data Science"]].mean(axis=1)
cleaned_df['Average'] = average
print(f"Highest average:{cleaned_df['Average'].max()}")
expression = cleaned_df['Average'] < 50
students_below_50 = cleaned_df.loc[expression,['Average']]
print(f"Students average below 50:{students_below_50}")
average = cleaned_df.mean(numeric_only=True)
print(f"Average mark each subject:{average}")
count_students_missing_subject = cleaned_df[["Maths","Python","Data Science"]].isna().sum()
print(f"count of Students missing subject marks:{count_students_missing_subject}")

cleaned_df.to_csv('cleaned_students.csv',index=False)

test_df = pd.read_csv('cleaned_students.csv')
print(test_df.head(7))

#final report
print("=================FINAL REPORT======================")
print(f"Total number of students:{test_df['Student'].size}")
print(f"==== AVERAGE MARK FOR EACH SUBJECT======")
print(test_df.mean(numeric_only=True))
highest_avg = test_df['Average'].max()
find = test_df['Average'] ==  highest_avg
highest_performer = test_df.loc[find] 
print("========== BEST PERFORMER ====================")
print(f"Highest-performing-student:{highest_performer}")
print("========== STUDENTS WHO NEED HELP ====================")
condi = test_df["Average"] < 50
low_performers = test_df.loc[condi]
print(f"Low performing students:{low_performers}")
print("========== STUDENTS MISSING MARKS ====================")
student_missing_marks = cleaned_df[["Maths","Python","Data Science"]].isna().sum(axis=1)
boolean_condition = student_missing_marks > 0
print(f"count of Students missing subject marks:{boolean_condition.sum()}")
print("=======================END=========================")

top_maths = cleaned_df.sort_values("Maths",ascending=False)
print(top_maths)

top_students = cleaned_df.sort_values(['Average','Student'],ascending=[False,True])
print(top_students)



top_python = cleaned_df.sort_values("Python",ascending=False)
print(top_python.head(3))







