# -*- coding:utf-8 -*-
'''
说明：
三层BP网络
'''
import math;
import numpy as ny;

#sigmoid函数
def sigmoid(x):
	return 1/(1+math.exp(-x));
#矩阵sigmoid函数
def sigmoid_matrix(matrix):
	return 1/(1+ny.exp(-matrix));
#sigmoid的导函数
def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x));
#已知矩阵的sigmoid值，求矩阵sigmoid导函数：
def sigmoid_derivative_matrix(matrix):
	return matrix*(1-matrix);
#线性预处理函数
def linear_pretreat(y):
	col=y.shape[1];
	for i in range(0,col):
		ymin=ny.min(y[0:,i]);
		ymax=ny.max(y[0:,i]);
		if ymin==ymax:
			y[0:,i]=y[0:,i]/ymin;
		else:
			y[0:,i]=(y[0:,i]-ymin)/(ymax-ymin);
#三层BP
class BPNet(object):
	#初始化网络
	def __init__(self,inputNum,hideNum,outNum):
		#输入层节点数
		self.inputNum=inputNum;
		#隐藏层节点数
		self.hideNum=hideNum;
		#输出层节点数
		self.outNum=outNum;

	#设置训练样本
	def set_sample(self,x,y):
		self.x=x;
		self.y=y;
	#设置权重
	#输入层到隐层，w[inputNum,hideNum]
	#隐层到输出层，w[hideNum,outNum]	
	def set_weight(self,ihWeight,hoWeight):
		#设置输入层到隐层的权重
		self.ihWeight=ihWeight;
		#设置隐层到输出层的权重
		self.hoWeight=hoWeight;

	#设置偏置
	#输入层到隐层，b[1,hideNum]
	#隐层到输出层，b[1,outNum]
	def set_bias(self,ihBias,hoBias):
		#设置输入层到隐层的偏置
		self.ihBias=ihBias;
		#设置隐层到输出层的偏置
		self.hoBias=hoBias;

	#设置学习率、最大误差、最大迭代次数
	def set_param(self,ihWeightRate,ihBiasRate,hoWeightRate,hoBiasRate,
		maxError,maxSteps):
		self.ihWeightRate=ihWeightRate;
		self.ihBiasRate=ihBiasRate;
		self.hoWeightRate=hoWeightRate;
		self.hoBiasRate=hoBiasRate;
		self.maxError=maxError;
		self.maxSteps=maxSteps;

	#设置激活函数及其导数，设置预处理函数
	def set_func(self,preFunc=linear_pretreat,actFunc=sigmoid_matrix,actDeri=sigmoid_derivative_matrix):
		#预处理函数
		self.preFunc=preFunc;
		#激活函数(矩阵)
		self.actFunc=actFunc;
		#激活函数导函数(矩阵)
		self.actDeri=actDeri;

	#训练网络
	def train_net(self,isBatch=True):
		#预处理y
		self.preFunc(self.y);
		#批处理(batch)
		if isBatch==True:
			for k in range(0,self.maxSteps):
				self.__forward_propagate(self.x);
				self.__back_propagate(self.x,self.y);
				self.errMean=ny.mean(ny.abs(output-self.y));
				if self.errMean<=self.maxError:
					print "error:%f" %self.errMean;
					break;
		#随机处理(stochastic)
		else:
			output=ny.zeros(self.y.shape);
			sampleNum=self.x.shape[0];
			for k in range(0,self.maxSteps):
				for i in range(0,sampleNum):
					self.__forward_propagate(self.x[i,0:]);
					self.__back_propagate(self.x[i,0:],self.y[i,0:]);
					output[i,0:]=self.output[0,0:];
				self.errMean=ny.mean(ny.abs(output-self.y));
				if self.errMean<=self.maxError:
					print "error:%f" %self.errMean;
					break;
					
	#测试数据
	def test_net(self,testData):
		self.__forward_propagate(sampleX=testData);
		print 'test data output:';
		print self.output;

	#写入文件
	def write_to_file(self,yFile='y.txt',outputFile='output.txt'):
		workdir=os.getcwd();
		ypath=workdir+'\\'+yFile;
		outpath=workdir+'\\'+outputFile;
		self.__forward_propagate(sampleX=self.x);
		with open(ypath,'w') as wf:
			ystr=str(self.y);
			ystr=ystr.replace('[',' ');
			ystr=ystr.replace(']',' ');
			wf.write(ystr);
		with open(outpath,'w') as wf:
			outstr=str(self.output);
			outstr=outstr.replace('[',' ');
			outstr=outstr.replace(']',' ');
			wf.write(outstr);
	#计算输出与样本误差平方和
	def __cal_err_square(self,output,y):
		errM=ny.matrix(output)-ny.matrix(self.y);
		errSquare=ny.sum(ny.array(errM)**2);
		errSquare/=2*errM.shape[0];
		return errSquare;
	#扩展偏置项为对应矩阵
	def __expand_bias(self,m):
		ihBias=self.ihBias;
		hoBias=self.hoBias;

		for i in range(1,m):
			ihBias=ny.row_stack((ihBias,self.ihBias));
			hoBias=ny.row_stack((hoBias,self.hoBias));
		return ny.array(ny.matrix(ihBias)),ny.array(ny.matrix(hoBias));

	#前向传播
	def __forward_propagate(self,sampleX):
		x=ny.array(ny.matrix(sampleX));
		ihBias,hoBias=self.__expand_bias(x.shape[0]);
		#计算隐层输出
		self.hideput=ny.dot(sampleX,self.ihWeight)+ihBias;
		self.hideput=self.actFunc(self.hideput);
		#计算输出层输出
		self.output=ny.dot(self.hideput,self.hoWeight)+hoBias;
		self.output=self.actFunc(self.output);
	
		
	#误差后向传播
	def __back_propagate(self,sampleX,sampleY):
		x=ny.array(ny.matrix(sampleX));
		y=ny.array(ny.matrix(sampleY));
		ihBias,hoBias=self.__expand_bias(sampleX.shape[0]);
		#计算输出层与隐层的导数
		outDeri=self.actDeri(self.output);
		hideDeri=self.actDeri(self.hideput);
		#输出层与样本的误差
		oyErr=(self.output-y)*outDeri;
		
		#隐层的误差
		hoErr=ny.dot(oyErr,self.hoWeight.T)
		hoErr=hoErr*hideDeri;
		
		#校正输入层到隐层的权重和偏置
		self.ihWeight-=self.ihWeightRate*ny.dot(x.T,hoErr);
		ihBias-=self.ihBiasRate*hoErr;
		self.ihBias=ny.array(ihBias[0]);
		#校正隐层到输出层的权重和偏置
		self.hoWeight-=self.hoWeightRate*ny.dot(self.hideput.T,oyErr);
		hoBias-=self.hoBiasRate*oyErr;
		self.hoBias=ny.array(hoBias[0]);
		
	


