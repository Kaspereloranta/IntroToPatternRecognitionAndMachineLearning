import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

# Print
print(f'Before sorting {my_numbers}')

# My sorting (e.g. bubble sort)

i = 1
while i < len(my_numbers):
    j = i
    while my_numbers[j] < my_numbers[j-1] and j > 0:
        cont1 = my_numbers[j]
        cont2 = my_numbers[j-1]
        my_numbers[j - 1] = cont1
        my_numbers[j] = cont2
        j = j - 1
    i = i + 1

# ADD HERE YOUR CODE

# Print
print(f'After sorting {my_numbers}')