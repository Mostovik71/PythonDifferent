import torch
from torch import nn
class Person:
    def __init__(self,name,surname):
        self.name=name
        self.surname=surname
    def print_info(self,n):
        for i in range(n):
         print(self.name,self.surname)
p1=Person('Vlad','Mostovik')
p1.print_info(2)

