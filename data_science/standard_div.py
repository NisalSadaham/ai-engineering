import numpy as np

A = [48, 49, 50, 51, 52]
B = [20, 35, 50, 65, 80]
C = [10,10,10,10,10,10]

print(np.std(A))
print(np.std(B))
print(np.std(C))


hours = [1, 2, 3, 4, 5]
marks = [40, 50, 60, 70, 80]

#calculating correlation using numpy(correlation is  -1 or 1 or 0)
#-1 = negative correlation
#1 = positive correlation
# 0 = non linear correlation
correlation = np.corrcoef(hours, marks)[0, 1]
print(correlation)

hours = [80, 70, 60, 50, 40]
correlation = np.corrcoef(hours,marks)[0,1]
print(correlation)

groupB = [20,35,50,65,80]
variance = np.var(groupB)
standard_deviation = np.std(groupB)

print("Variance:",variance)
print("Standard deviation:",standard_deviation)

marks = [60, 70, 80, 90]
variance = np.var(marks,ddof = 1)
standard_deviation = np.std(marks,ddof=1)

print("Variance:",variance)
print("Standard deviation:",standard_deviation)



marks = [10, 20, 30, 40, 50, 60, 70, 80]

Q1 =np.percentile(marks,25)
Q3 = np.percentile(marks,75)

IQR = Q3-Q1

lower_bound = Q1 - (1.5 *IQR)
higher_bound = Q3 + (1.5*IQR)


print(IQR)
print(lower_bound)
print(higher_bound)
