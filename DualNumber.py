"""
@author Antonio Bruce Webb(Organi)
"""
from math import pow


class DualNumber:

    def __init__(self, real, dual):
        self.r = real
        self.d = dual

    def __add__(self, dual_num):
        return self.add(dual_num)

    def __pow__(self, power, modulo=None):
        return self.pow(power)

    def __sub__(self, dual_num):
        return self.sub(dual_num)

    def __mul__(self, dual_num):
        return self.multiply(dual_num)

    def __truediv__(self, dual_num):
        return self.divide(dual_num)

    def __floordiv__(self, dual_num):
        return self.divide(dual_num)

    def real(self):
        return self.r

    def dual(self):
        return self.d

    def add(self, dual_num):
        new_real = self.r + dual_num.real()
        new_dual = self.d + dual_num.dual()
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def sub(self, dual_num):
        new_real = self.r - dual_num.real()
        new_dual = self.d - dual_num.dual()
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def divide(self, dual_num):
        new_real = self.r / dual_num.real()
        new_dual = ((dual_num.real() * self.d) - (self.r * dual_num.dual())) / dual_num.real() ** 2
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def multiply(self, dual_num):
        new_real = self.r * dual_num.real()
        new_dual = (self.r * dual_num.dual()) + (dual_num.real() * self.d)
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def dual_pow_dual(self, dual_num):
        new_real = pow(self.r, dual_num.real())
        new_dual = (new_real * dual_num.dual() * log(self.r)) + ((new_real * self.d * dual_num.real()) / self.r)
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def pow(self, power):
        new_dual_num = DualNumber(self.r, self.d)
        dual_one = DualNumber(1, 0)
        for inter in range(abs(power)):
            new_dual_num = new_dual_num * self
        if power == 0:
            new_dual_num = dual_one
        elif power < 0:
            new_dual_num = dual_one / new_dual_num
        return new_dual_num

    def nth_root(self, n):
        new_real = pow(self.r, 1/n)
        new_dual = (new_real * self.d) / (2 * self.r)
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

