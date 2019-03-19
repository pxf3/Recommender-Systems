a = [1, 1]

aSum = sum(a)
for i in range(0, len(a)):
    a[i] = a[i]/aSum

print(a)
