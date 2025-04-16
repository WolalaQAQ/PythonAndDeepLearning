class Student:
    def __init__(self, name, age, gender, address, phone):
        self.name = name
        self.age = age
        self.gender = gender
        self.address = address
        self.phone = phone

    def eat(self):
        print(f"{self.name} is eating.")

    def study(self):
        print(f"{self.name} is studying.")

    def play(self):
        print(f"{self.name} is playing.")


if __name__ == '__main__':
    stu = Student('张三', 20, '男', '辽宁省沈阳市', '12345678901')
    print(f"姓名: {stu.name}, 年龄: {stu.age}, 性别: {stu.gender}, 地址: {stu.address}, 电话: {stu.phone}")
    stu.eat()
    stu.study()
    stu.play()

