from abc import ABCMeta, abstractmethod


class Object(object):
    __metaclass__ = ABCMeta
    def __init__(self, a):
        self.a = a

    @abstractmethod
    def foo(self, x):
        return 0



class CObject(Object):
    def __init__(self, a, b):
        super().__init__(a)
        self.b = b



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