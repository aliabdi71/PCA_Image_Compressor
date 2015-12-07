from PIL import Image
import numpy as np
import sklearn.decomposition as deco
import pylab
import threading

def matrixMul(X, Y):
	result = np.zeros((len(X), len(Y[0])))
	t1 = threading.Thread(target=calculator, args = (X,Y,result,0))
	t1.start()
	t2 = threading.Thread(target=calculator, args = (X,Y,result,16))
	t2.start()
	t3 = threading.Thread(target=calculator, args = (X,Y,result,32))
	t3.start()
	t4 = threading.Thread(target=calculator, args = (X,Y,result,48))
	t4.start()
	t1.join()
	t2.join()
	t3.join()
	t4.join()
	print(result)
	return result
	
def calculator(X, Y, result, i):
	for m in range(i, i+16):
		for j in range(len(Y[0])):
			for k in range(len(Y)):
			   result[m][j] += X[m][k] * Y[k][j]
		print(m)

im_original = np.asarray(Image.open('1.png').convert('L'))				#converting image into N*M matrix

linear_matrix=[]

height = len(im_original)
width = len(im_original[0])
for i in range(0, height/8):
	for j in range(0, width/8):
		index_i = i*8
		index_j = j*8
		linear_matrix.append(im_original[index_i:index_i+8, index_j:index_j+8].reshape(1, 64))			#converting image matrix into linear_matrix
x = []
for i in range(0, len(linear_matrix)):
	x.append(linear_matrix[i][0])

std_x = np.std(x, 0)
mean_x = np.mean(x, axis=0)
x = (x - np.mean(x, axis=0)) / np.std(x, 0) # You need to normalize your data first
cov = np.ma.cov(x, rowvar=False)

D, V = np.linalg.eig(cov)

for k in (5, 10, 20, 30, 40, 50, 60):
	tV = np.transpose(V)
	x_new = matrixMul((tV / np.linalg.norm(tV)), np.transpose(x))
	for c in range(0, k):
		for r in range(0, 64):
			V[r][c] = 0
	x_Ret = matrixMul((V / np.linalg.norm(V)), x_new)
	x_Ret = np.transpose(x_Ret)
	x_Ret = (x_Ret * std_x) + mean_x
	x_Ret = x_Ret.reshape(512, 512)
	pylab.figure()
	pylab.gray()
	pylab.imshow(x_Ret)
	pylab.show()
	pylab.savefig('converted//' + str(k) + '.png')
	pylab.clf()