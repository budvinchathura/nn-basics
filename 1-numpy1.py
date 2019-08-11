print("hello")
import numpy as np

arr1 = [1,2,3,4]
print(arr1)

narr1 = np.array(arr1)          #normal array to numpy array
print(type(narr1))

print(narr1)
print(narr1.shape)              #dimension of the numpy array as a tuple

narr2 = np.zeros((2,2,2))       #fill all zeros according to the given dimension
print(narr2)
print(narr2.shape)

narr3 = np.full((2,2,1),5)      #fill all with the given value
print(narr3)

narr4 = np.eye(3)               #2d array with diagonal values=1 others=0
print(narr4)

narr5 = np.eye(5,k=1)           #same as above but diagonal is shifted
print(narr5)

narr6 = np.eye(6,k=-2)
print(narr6)