import numpy as np

sample = np.array([20,30,40,50])
print(sample)
print(type(sample))
print(sample + 5)
print(sample * 2)
print(sample.max())
print(sample.min())
print(sample.mean())

marks = np.array([72, 88, 56, 91, 63])
print(marks[:3])
print(marks.shape)
print(marks.size)
print(marks.ndim)

data = np.array([
    [5, 10],
    [15, 20],
    [25, 30]
])

data[1,1] = 100
print(data[1,1])
print(data.argmin())#index of the smallest value
print(data.argmax())#index of largest value
print(data.max(axis=0))#axis finds maximum in every column
#axis=1 for row
print(data.sum()) #gets the sum of a array
result = np.where(marks < 50, 0, marks) #conditions other way
#np.where(condition, value_if_true, value_if_false)
#print(np.std(a)) standard deviation
marks = np.array([35, 48, 55, 67, 82, 91, 44, 76])

min_passing_mark = np.min((marks[marks >= 50]))
max_failed_mark = np.max((marks[marks < 50]))
print(min_passing_mark)
print(max_failed_mark)
average_passed = np.sum(np.where(marks >= 50,marks,0)) / np.sum(marks >= 50)
print(average_passed)
