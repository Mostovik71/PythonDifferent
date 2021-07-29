from numba import cuda,jit
import time

import numpy as np
start1=time.time()
@jit
def summa1(a):
 for i in range(a):
     a=a+1
 print(a)
summa1(1000000000)
end1=time.time()
time1=end1-start1
print(time1)
