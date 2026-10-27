import numpy as np

marks = np.array([
    [78, 65, 82],
    [45, 38, 55],
    [90, 88, 95],
    [32, 48, 40],
    [67, 72, 60],
    [85, 91, 78]
])

subjects = np.array(["Python", "Maths", "Data Science"])

shape = marks.shape
dimensions = marks.ndim
total_elements = marks.size

print(f"Shape of marks:{shape}")
print(f"Number of dimenstions:{dimensions}")
print(f"Total number of elements:{total_elements}")

student_average = np.mean(marks,axis=1)
print(f"Student average{student_average}")
highest_scoring_student_index = np.argmax(student_average)
print(f"Highest scoring student index:{highest_scoring_student_index}")
top_student_marks = marks[highest_scoring_student_index]
print(f"Top student marks:{top_student_marks}")

students_needing_help = student_average < 50
print(f"Students who needs help:{students_needing_help}")
help_needed_averages = student_average[students_needing_help]
print(f"Help needed averages:{help_needed_averages}")
students_needing_help_count = np.sum(student_average < 50)
print(f"No of students who needs help:{students_needing_help_count}")

improved_marks = marks.copy()

condtion = improved_marks < 40

improved_marks[condtion] += 5

print(improved_marks)

rng_1 =np.random.default_rng(42)
random_marks = rng_1.integers(1,101,10)
print(random_marks)