import numpy as np

arr1 = [[1,2,3,4],[5,6,7,8],[0,0,0,0]]
narr1 = np.array(arr1)
print(narr1)
sum1 = narr1.sum()              #sum of al values
print(sum1)
print(narr1.sum(axis=0))        #column wise sum
print(narr1.sum(axis=1))        #row wise sum
print(narr1.mean(axis=0))        #column wise mean
print(narr1.mean(axis=1))        #row wise mean
print(narr1.mean())        #overall mean

print(np.median(narr1,axis =0))        #column wise median
print(np.median(narr1,axis = 1))        #row wise median

print(np.percentile(narr1,50));

