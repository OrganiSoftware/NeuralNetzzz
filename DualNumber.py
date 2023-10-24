"""
@author Antonio Bruce Webb(Organi)
"""
import math


class DualNumber:

    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def real(self):
        return self,real

    def dual(self):
        return self.dual

    def devide(self, dual_num):
        new_real = self.real / dual_num.real()
        new_dual = ((dual_num.real() * self.dual) - (self.real * dual_num.dual()))/dual_num.real()**2
        self.real = new_real
        self.dual = new_dual

    def multiply(self, dual_num):
        new_real = self.real * dual_num.real()
        new_dual = (self.real * dual_num.dual()) + (dual_num.real() * self.dual)
        self.real = new_real
        self.dual = new_dual

    def pow(self, pow):
        if not pow == 0:
            count = abs(pow)
            while(count > 0):
                if (pow > 0 and count > 1):
                    self.multiply(self)
                elif (pow < 0 and count > 0):
                    self.devide(self)
        else:
            self.real = 1
            self.dual = 0

    def nth_root(self, n):
        new_real = math.pow(self.real, 1/n)
        new_dual = (new_real * self.dual) / (2 * self.real)
        self.real = new_real
        self.dual = new_dual

