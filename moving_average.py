"""
Interview question at Barclays

Write an efficient rolling window function. 

A naive approach is to add the last 10 numbers but the complexity
for each iteration is in the order of N.  The more efficient approach
is to each time that you read a number you substract a number.

"""
def naive(v: list[float], N: int):
    return [sum(v[i: i + N]) / N for i in range(len(v) - N + 1)] 


def circular_buffer(v: list[float], N: int):
    window = [0 for i in range(N - 1)]
    total = 0
    for i, a in enumerate(v):
        j = i % (N - 1)
        first = window[j]
        total += a
        window[j] = a 
        if i >= N -1: 
            yield total / N
        total -= first

for i, f in enumerate((naive, circular_buffer)):
    res = list(f(range(8), 4))
    print(f"asserting {i}, {res}")
    assert [1.5, 2.5, 3.5, 4.5, 5.5] == res



