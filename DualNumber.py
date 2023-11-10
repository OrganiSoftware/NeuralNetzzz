"""
@author Antonio Bruce Webb(Organi)
"""
from math import pow
class DualNumber:

    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

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
        return self.real

    def dual(self):
        return self.dual

    def add(self, dual_num):
        new_real = self.real + dual_num.real
        new_dual = self.dual + dual_num.dual
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def sub(self, dual_num):
        new_real = self.real - dual_num.real
        new_dual = self.dual - dual_num.dual
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def divide(self, dual_num):
        new_real = self.real / dual_num.real()
        new_dual = ((dual_num.real() * self.dual) - (self.real * dual_num.dual())) / dual_num.real() ** 2
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def multiply(self, dual_num):
        new_real = self.real * dual_num.real()
        new_dual = (self.real * dual_num.dual()) + (dual_num.real() * self.dual)
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def pow(self, power):
        new_real = 0
        new_dual = 0
        if power > 0:
            new_real = pow(self.real, power)
            new_dual = power * pow(self.real, power - 1) * self.dual
        elif power < 0:
            new_real = (pow(self.real, power) / pow(self.real, 2 * abs(power)))
            new_dual = -((2 * self.real * self.dual) / pow(self.real, 2 * abs(power)))
        else:
            new_real = 1
            new_dual = 0
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

    def nth_root(self, n):
        new_real = pow(self.real, 1/n)
        new_dual = (new_real * self.dual) / (2 * self.real)
        new_dual_num = DualNumber(new_real, new_dual)
        return new_dual_num

