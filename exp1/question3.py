# 判断是否是素数
def prime(n):
    if n <= 1:
        raise ValueError("Input must be a positive integer greater than 1")
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    num = int(input("Enter a number: "))
    if prime(num):
        print(f"{num} is a prime number.")
    else:
        print(f"{num} is not a prime number.")
