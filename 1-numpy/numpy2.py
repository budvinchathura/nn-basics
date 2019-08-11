import numpy as np
arr1 = [[1,2],[3,4,5],[8]]      #this is not suitable
narr1 = np.array(arr1);
print(narr1)

arr2 = [[1,2,2],[3,4,5],[7,7,7]]
narr2 = np.array(arr2);
print(narr2);

narr3 = np.array(arr2,dtype = float)        #data type = float
print(narr3)  

#these two are same
print(narr3[1][2])          #5.0
print(narr3[1,2])          #5.0

narr3[1,2] = 4.9        #change the value
print(narr3)


x = [1,2,3]
y = [2,4,6]

xy = np.concatenate((x,y))      #just join them
print(xy)

narr4 = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
print(narr4)
print(narr4[1:3,3:5])           #selected region
print(narr4[3:5,2:5])           #another selected region