import numpy as np

no_stu_passed = 15
total_students = 20

probabiliy = no_stu_passed/total_students

print(probabiliy)

results = ["Pass", "Fail", "Pass", "Pass", "Fail",
           "Pass", "Pass", "Fail", "Pass", "Pass"]
passed = results.count("Pass")
failed = results.count("Fail")
total = len(results)

probability_passed = passed/total
probability_failed = failed/total

print(probability_passed)
print(probability_failed)

sample_table = 1
probability_pass = 0.82

probability_fail = sample_table - probability_pass

print(probability_fail)

event_a = 0.8
event_b = 0.5

both_happening = event_a * event_b
print(both_happening)

neither_happening = (1 - 0.8) * (1 - 0.5)
print(neither_happening)
at_least_one = 1 - neither_happening
print(at_least_one)

python_lovers = 18
both_languages = 8

prob = both_languages/python_lovers

print(prob)