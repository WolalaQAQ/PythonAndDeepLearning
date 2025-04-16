sum = 0
for i in range(1, 100):
    if i % 2 == 0:
        sum -= i**2
    else:
        sum += i**2

print(f"The sum is: {sum}")
