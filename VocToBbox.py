# -*- coding:utf-8 -*-
import json
import os 
from os import listdir, getcwd
from os.path import join
import os.path
import glob
rootdir='./label_json_visible/'#写自己存json的数据地址
labeldir='./2/'
def position(pos):#该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x=[]
    y=[]
    nums=len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max=max(x)
    x_min=min(x)
    y_max=max(y)
    y_min=min(y)
    b=(float(x_min),float(x_max),float(y_min),float(y_max))
    return b
 
def convert(size, box):#该函数将xmin,ymin,xmax,ymax转为x,y,w,h中心点坐标和宽高
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(image_id):   
    load_f=open(rootdir+"%s.json"%(image_id),'r')#导入json标签的地址
    load_dict = json.load(load_f)
    out_file = open(labeldir+'%s.txt'%(image_id), 'w')#输出标签的地址
    #keys=tuple(load_dict.keys()) 
    w=load_dict['imageWidth']#原图的宽，用于归一化
    h=load_dict['imageHeight']
    #print(h)
    objects=load_dict['shapes']
    nums=len(objects)
    #print(nums)
    #object_key=tuple(objects.keys()
   
    
    
    for i in range(0,nums):
        labels=objects[i]['label']
        #print(i)
        if (labels in ['pedestrian']):
            #print(labels)
            pos=objects[i]['points']   
            b=position(pos)
            bb = convert((w,h), b)
            cls_id=0
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            #print(type(pos))
        elif (labels in ['cyclist']):
            #print(labels)
            pos=objects[i]['points']
            b=position(pos)
            bb = convert((w,h), b)
            cls_id=1
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        elif (labels in ['car']):
            #print(labels)
            pos=objects[i]['points']
            b=position(pos)
            bb = convert((w,h), b)
            cls_id=2
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        elif (labels in ['bus']):
            #print(labels)
            pos=objects[i]['points']
            b=position(pos)
            bb = convert((w,h), b)
            cls_id=3
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        elif (labels in ['truck']):
            #print(labels)
            pos=objects[i]['points']
            b=position(pos)
            bb = convert((w,h), b)
            cls_id=4
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        elif (labels in ['traffic_light']):
            #print(labels)
            pos=objects[i]['points']
            b=position(pos)
            bb = convert((w,h), b)
            cls_id=5
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        elif (labels in ['traffic_sign']):
            #print(labels)
            pos=objects[i]['points']
            b=position(pos)
            bb = convert((w,h), b)
            cls_id=6
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
def image_id(rootdir):
    a=[]
    paths = glob.glob(os.path.join(rootdir, '*.json'))
    paths.sort()
    #print(paths)
    for filename in paths:
            filename=filename.split(rootdir)[1].split('.json')[0]
            print(filename)
            a.append(filename)
    return a
names=image_id(rootdir)
for image_id in names:
    print('image_id:',image_id)
    convert_annotation(image_id)
