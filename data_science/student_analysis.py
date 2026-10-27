import numpy as np
#array of marks
marks = np.array([35,48,67,55,82,91,44,76])
#function to show overall statistics
def overall_statistics(marks):
    print('=====OVERALL STATISTICS=====')
    print(f'Number of students:{np.size(marks)}')
    print(f'Highest mark:{np.max(marks)}')
    print(f'Lowest mark:{np.min(marks)}')
    print(f'Median mark:{np.median(marks)}')
    print(f'Standard Deviation:{np.std(marks)}')
    print(f'Range:{np.max(marks) - np.min(marks)}')
#function to analyse pass/fail
def pass_fail_analysis(marks):
    print('=====PASS/FAIL ANALYSIS=====')
    print(f'Passsed students:{np.sum(marks >= 50)}')
    print(f'Failed students:{np.sum(marks < 50)}')
    print(f'Pass percentage:{(np.sum(marks >= 50)/np.size(marks)) * 100}')
    print(f'fail percentage:{(np.sum(marks < 50)/np.size(marks)) * 100}')
    print(f"Sudent results:{np.where(marks >= 50,'Pass','Fail')}")
#function to analyse passed students
def passed_student_analysis(marks):
    print('=====PASSED STUDENTS ANALYSIS=====')
    print(f'Average passing mark:{marks[marks >= 50].mean()}')
    print(f'Highest passing mark:{marks[marks >= 50].max()}')
    print(f'Lowest passing mark:{marks[marks >= 50].min()}')
#function to analyse failed students
def failed_student_analysis(marks):
    print('=====FAILED STUDENTS ANALYSIS=====')
    print(f'Average failing mark:{marks[marks < 50].mean()}')
    print(f'Highest failing mark:{marks[marks < 50].max()}')
    print(f'Lowest failing mark:{marks[marks < 50].min()}')
#function to identify students who needs help
def help_needed_students(marks):
    print('=====HELP NEEDED STUDENTS ANALYSIS=====')
    print(f'Marks Below 50:{marks[marks < 50]}')
    print(f'Marks Below 40:{marks[marks < 40]}')
    print(f'Marks Below 40:{np.sum(marks < 40)}')
#function to identify to performing students
def performing_students(marks):
    print('=====TOP PERFORMING STUDENTS ANALYSIS=====')
    print(f'Highest mark:{marks.max()}')
    print(f'Index of highest mark:{np.argmax(marks)}')
    print(f'Marks of students scoring at least 80:{marks[marks >= 80]}')

overall_statistics(marks)
pass_fail_analysis(marks)
passed_student_analysis(marks)
failed_student_analysis(marks)
help_needed_students(marks)
performing_students(marks)
    