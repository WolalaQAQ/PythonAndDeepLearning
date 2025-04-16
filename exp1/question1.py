import numpy as np

radius = -1

radius = float(input("Enter the radius of the ball: "))
if radius < 0:
    raise ValueError("Radius cannot be negative")

volume = (4/3) * np.pi * radius**3

print(f"The volume of the ball is: {volume}")
