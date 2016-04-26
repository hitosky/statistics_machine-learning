# -*- coding:utf-8 -*-
'''
说明：
自回归模型建模
x[i]=a0+a1*x[i-1]+a2*x[i-2]+...+ai*x[0];
求解系数[a0,a1,......ai]
'''
import numpy;
import math;
#计算某个k值下的自相关系数
def auto_relate_coef(data,avg,s2,k):
	ef=0.;
	for i in range(0,len(data)-k):
		ef=ef+(data[i]-avg)*(data[i+k]-avg);
	ef=ef/len(data)/s2;
	return ef;
#计算自相关系数
def auto_relate_coefs(sample):
	efs=[];
	data=[];
	avg=numpy.mean(sample);
	s2=numpy.var(sample);
	array=sample.reshape(1,-1);
	for x in array.flat:
		data.append(x);
	for k in range(0,len(data)):
		ef=auto_relate_coef(data,avg,s2,k);
		efs.append(ef);
	return efs;

#最小二乘法计算AR(p)模型参数，返回系数数组和求得的回归输出
def ar_least_square(sample,p):
	matrix_x=numpy.zeros((sample.size-p,p));
	#matrix_x=matrix_x.reshape(sample.size-p,p);
	matrix_x=numpy.matrix(matrix_x);
	array=sample.reshape(sample.size);
	j=0;
	for i in range(0,sample.size-p):
		matrix_x[i,0:p]=array[j:j+p];
		j=j+1;
	matrix_y=numpy.array(array[p:sample.size]);
	matrix_y=matrix_y.reshape(sample.size-p,1);
	matrix_y=numpy.matrix(matrix_y);
	fi=numpy.dot(numpy.dot((numpy.dot(matrix_x.T,matrix_x)).I,matrix_x.T),matrix_y);
	fi=numpy.round(fi,3);
	matrix_y=numpy.dot(matrix_x,fi);
	matrix_y=numpy.row_stack((array[0:p].reshape(p,1),matrix_y));
	return fi,matrix_y;

#AIC法则计算p阶信息
def ar_aic(rss,p):
	n=rss.size;
	s2=numpy.var(rss);
	return 2*p+n*math.log(s2);
#SC法则计算p阶信息
def ar_sc(rss,p):
	n=rss.size;
	s2=numpy.var(rss);
	return p*math.log(n)+n*math.log(s2);
#模型的F检验
def ar_f_test(sample,ar_model):
	sample_s2=numpy.var(sample);
	ar_model_s2=numpy.var(ar_model);
	return sample_s2>ar_model_s2 and sample_s2/ar_model_s2 or ar_model_s2/sample_s2;