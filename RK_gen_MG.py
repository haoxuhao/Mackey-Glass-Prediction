import numpy as np 
from matplotlib import pyplot as plt
from math import floor
import os.path as osp
import os

def Df(x):
	a=0.2
	y=a*x/(1+x**10)
	return y

def Mackey_Glass(N, tau, time_interval=1):
	'''
	4阶龙格库塔计算MG序列
	参考：https://blog.csdn.net/ddpiccolo/article/details/89464435
	'''
	x = np.zeros((N,))
	t = np.zeros((N,))
	
	b=0.1
	h = time_interval
	x[0] = 1.2

	for k in range(N-1):	
		t[k+1] = t[k]+h
		if k < tau:
			k1=-b*x[k]
			k2=-b*(x[k]+h*k1/2); 
			k3=-b*(x[k]+k2*h/2); 
			k4=-b*(x[k]+k3*h);
			x[k+1]=x[k]+(k1+2*k2+2*k3+k4)*h/6; 
		else:
			n=floor((t[k]-tau-t[0])/h+1); 
			k1=Df(x[n])-b*x[k]; 
			k2=Df(x[n])-b*(x[k]+h*k1/2); 
			k3=Df(x[n])-b*(x[k]+2*k2*h/2); 
			k4=Df(x[n])-b*(x[k]+k3*h); 
			x[k+1]=x[k]+(k1+2*k2+2*k3+k4)*h/6; 

	return t, x


def test():
	t, x = Mackey_Glass(6000, 17)
	print(np.max(x), np.min(x))
	figure = plt.figure()
	plt.xlabel("time", fontsize=13)
	plt.ylabel("value", fontsize=13)
	plt.title("MG-series")
	plt.plot(t, x)
	plt.savefig("data/data.png")
	#plt.show()

if __name__=="__main__":
	#test()
	data_dir = "./data"
	if not osp.exists(data_dir):
		os.makedirs(data_dir)

	t, x = Mackey_Glass(6000, 17)
	np.save(osp.join(data_dir, "data.npy"), x)