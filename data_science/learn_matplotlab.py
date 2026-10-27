import matplotlib.pyplot as plt
#line charts
"""days = [1,2,3,4,5]
marks = [45, 55, 60, 72, 85]

plt.plot(days,marks,marker="o",linestyle = "--")
plt.title("Student Progress")
plt.xlabel("Day")
plt.ylabel("Marks")
plt.grid(visible=True)

plt.show()"""


#subjects = ["Python", "Maths", "Data Science"]
#marks = [78, 65, 82]

"""plt.bar(subjects,marks)
plt.title("Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Marks")
plt.grid(axis="y")
plt.show()"""
"""
hours = [1, 2, 3, 4, 5, 6]
marks = [45, 50, 58, 65, 72, 82]

plt.scatter(hours,marks)
plt.title("Study hours vs Marks")
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.grid(visible=True)
plt.show()
"""
marks = [45, 52, 55, 61, 64, 65, 67, 70, 72, 73,
         75, 76, 78, 80, 82, 85, 88, 90, 92, 95]

plt.hist(marks,bins=10)
plt.title("Distirbution of student marks")
plt.xlabel("Marks")
plt.ylabel("Number of students")
plt.grid(axis="y")
plt.show()
