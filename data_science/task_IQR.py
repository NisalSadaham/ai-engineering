import numpy as np

marks = [45, 50, 52, 48, 55, 51, 49, 53, 47, 95]

q1 = np.percentile(marks,25)
q3 = np.percentile(marks,75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
#outliear detector
for mark in marks:
    if mark < lower_bound or mark > upper_bound:
        print(f"Potential outlier found: {mark}")

