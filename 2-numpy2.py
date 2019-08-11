import numpy as np
arr1 = [[1,2],[3,4,5],[8]]
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