# -*- coding:utf-8 -*-
'''
说明：
聚类算法，基于：
1.K-means
'''

import numpy;

#判断data属于哪个聚类簇
def judge_cluster(obj,data):
	j=0;
	temp=abs(data-obj[0]);
	for i in range(1,len(obj)):
		if abs(data-obj[i])<temp:
			j=i;
			temp=abs(data-obj[i]);
	return j;
#计算平方误差和
def cal_square_error(obj,clusters):
	err=0.;
	for i in range(0,len(obj)):
		for data in clusters[i]:
			err+=(data-obj[i])**2;
	return err;
#计算簇间最大距离
def clusters_max_distance(obj):
	temp=0.;
	for o1 in obj:
		for o2 in obj:
			if abs(o1-o2)>temp:
				temp=abs(o1-o2);
	return temp;
#计算每个簇的最大半径
def clusters_rad(obj,clusters):
	rad=[];
	i=0;
	for cluster in clusters:
		temp=0.;
		for c in cluster:
			if abs(c-obj[i])>temp:
				temp=abs(c-obj[i]);
		rad.append(temp);
		i+=1;
	return rad;

#k_means方法生成聚类簇，obj为簇中心集合，maxk为最大迭代次数，error为允许最大精度
def k_means(obj,trainData,maxk,error):
	clusters=[];
	for i in range(0,len(obj)):
		clusters.append([]);
	for data in trainData:
		j=judge_cluster(obj,data);
		clusters[j].append(data);
	squareErr=cal_square_error(obj,clusters);

	for k in range(0,maxk):
		for i in range(0,len(obj)):
			obj[i]=numpy.mean(numpy.array(clusters[i]));
		for i in range(0,len(obj)):
			clusters[i]=[];
		for data in trainData:
			j=judge_cluster(obj,data);
			clusters[j].append(data);
		err=cal_square_error(obj,clusters);
		if abs(squareErr-err)<=error:
			break;
		squareErr=err;
	return clusters;



