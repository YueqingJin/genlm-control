
def even_bit_set_number(n: int) -> int:
    count = 0
    res = n
    temp = n
    while temp > 0:
        if count % 2 == 1:
            res |= (1 << count)
        count += 1
        temp >>= 1
    return res