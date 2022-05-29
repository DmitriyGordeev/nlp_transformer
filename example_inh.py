
class Object:
    def __init__(self, a):
        self.a = a

    def foo(self, x):
        return 0



class CObject(Object):
    def __init__(self, a, b):
        super().__init__(a)
        self.b = b

    def foo(self, x):
        print(self.a, ", arg:", x)


class User:
    def __init__(self, age):
        self.age = age

    def use_object(self, obj: Object):
        r = obj.foo(self.age)
        pass


def main():
    obj = Object(19)
    user = User(20)
    user.use_object(obj)


if __name__ == "__main__":
    main()