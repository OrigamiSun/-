import base64
import json
from labelme import utils
import cv2 as cv
import sys
import numpy as np
import random
import re
from PIL import Image,ImageEnhance, ImageOps  
import math
import os
import glob

def DataAugment(num_agu, image_path, image_id, savepath,image_type,label_path,cases,save_label_path,n):
    #def __init__(self, num_agu, image_path, image_id, image_savepath,image_type):  ## 翻转运动
        print('path=====',image_id)
        '''cases=random.sample(range(1, 65), num_agu)  ## 产生不重复随机整数
        #cases=[4]
        print('cases=====',cases)'''
        
        s_path=str(image_path)+'/'+str(image_id)+str(image_type)

        print('s_path=====',s_path)

        img = cv.imread(s_path)

        (h, w) = img.shape[:2]
           
        try:
            img.shape
        except:
            print('No Such image!---'+str(id)+'.jpg')
            sys.exit(0)
 ################ 翻转变换 ######################################################
        dst1 = cv.flip(img, 0, dst=None) ## 沿着x轴翻转
        dst2 = cv.flip(img, 1, dst=None) ## 沿着y轴翻转
        dst3 = cv.flip(img, -1, dst=None) ## 沿着x和y轴同时翻转
        flip_x = dst1
        flip_y = dst2
        flip_x_y = dst3
 ################ 旋转变换 ######################################################
        #M = cv.getRotationMatrix2D((w // 2, h // 2), 45, 1.0) ## 旋转45度 
        #dst4 = cv.warpAffine(img, M, (w, h))
        #M1 = cv.getRotationMatrix2D((w // 2, h // 2), 90, 1.0) ## 旋转90度 
        #dst5 = cv.warpAffine(img, M1, (w, h))
        #M2 = cv.getRotationMatrix2D((w // 2, h // 2), 135, 1.0) ## 旋转135度 
        #dst6 = cv.warpAffine(img, M2, (w, h))
        #M3 = cv.getRotationMatrix2D((w // 2, h // 2), 225, 1.0) ## 旋转225度 
        #dst7 = cv.warpAffine(img, M3, (w, h))
        #M4 = cv.getRotationMatrix2D((w // 2, h // 2), 270, 1.0) ## 旋转270度 
        #dst8 = cv.warpAffine(img, M4, (w, h))
        #M5 = cv.getRotationMatrix2D((w // 2, h // 2), 315, 1.0) ## 旋转315度 
        #dst9 = cv.warpAffine(img, M5, (w, h))
        #rote_45 = dst4
        #rote_90 = dst5
        #rote_135 = dst6
        #rote_225 = dst7
        #rote_270 = dst8
        #rote_315 = dst9

######################### 求旋转矩阵 #######################################################
        ############### 旋转45 ######################
        mat_45 = cv.getRotationMatrix2D((w//2, h//2), -45, 1)
        Rot_45 = cv.invertAffineTransform(mat_45)
        vertex_45 =np.dot(Rot_45,np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]])) # 求得变换后的坐标
        bian_45=max(int((max(vertex_45[0])-min(vertex_45[0]))),int((max(vertex_45[1])-min(vertex_45[1]))))
        padding_h_45 = int((bian_45-h) // 2)
        padding_w_45 = int(( bian_45-w) // 2)
        center_45 = ( bian_45 // 2,  bian_45 // 2)
        img_padded_45 = np.zeros(shape=( bian_45,  bian_45, 3), dtype=np.uint8)
        img_padded_45[padding_h_45:padding_h_45+h, padding_w_45:padding_w_45+w, :] = img
        M_45 = cv.getRotationMatrix2D(center_45, 45, 1)
        dst4 = cv.warpAffine(img_padded_45, M_45, ( bian_45,  bian_45))
        rote_45 = dst4
        ############### 旋转90 ######################
        mat_90 = cv.getRotationMatrix2D((w//2, h//2), -90, 1)
        Rot_90 = cv.invertAffineTransform(mat_90)
        vertex_90 = np.dot(Rot_90,np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]])) # 求得变换后的坐标
        bian_90 = max(int((max(vertex_90[0])-min(vertex_90[0]))),int((max(vertex_90[1])-min(vertex_90[1]))))
        padding_h_90 = int((bian_90-h) // 2)
        padding_w_90 = int(( bian_90-w) // 2)
        center_90 = ( bian_90 // 2,  bian_90 // 2)
        img_padded_90 = np.zeros(shape=( bian_90,  bian_90, 3), dtype=np.uint8)
        img_padded_90[padding_h_90:padding_h_90+h, padding_w_90:padding_w_90+w, :] = img
        M_90 = cv.getRotationMatrix2D(center_90, 90, 1)
        dst5 = cv.warpAffine(img_padded_90, M_90, ( bian_90,  bian_90))
        rote_90 = dst5
        ############### 旋转135 ######################
        mat_135 = cv.getRotationMatrix2D((w//2, h//2), -135, 1)
        Rot_135 = cv.invertAffineTransform(mat_135)
        vertex_135 =np.dot(Rot_135,np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]])) # 求得变换后的坐标
        bian_135=max(int((max(vertex_135[0])-min(vertex_135[0]))),int((max(vertex_135[1])-min(vertex_135[1]))))
        padding_h_135 = int((bian_135-h) // 2)
        padding_w_135 = int(( bian_135-w) // 2)
        center_135 = ( bian_135 // 2,  bian_135 // 2)
        img_padded_135 = np.zeros(shape=( bian_135,  bian_135, 3), dtype=np.uint8)
        img_padded_135[padding_h_135:padding_h_135+h, padding_w_135:padding_w_135+w, :] = img
        M_135 = cv.getRotationMatrix2D(center_135, 135, 1)
        dst6 = cv.warpAffine(img_padded_135, M_135, ( bian_135,  bian_135))
        rote_135 = dst6
        ############### 旋转225 ######################
        mat_225 = cv.getRotationMatrix2D((w//2, h//2), -225, 1)
        Rot_225 = cv.invertAffineTransform(mat_225)
        vertex_225 =np.dot(Rot_225,np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]])) # 求得变换后的坐标
        bian_225=max(int((max(vertex_225[0])-min(vertex_225[0]))),int((max(vertex_225[1])-min(vertex_225[1]))))
        padding_h_225 = int((bian_225-h) // 2)
        padding_w_225 = int(( bian_225-w) // 2)
        center_225 = ( bian_225 // 2,  bian_225 // 2)
        img_padded_225 = np.zeros(shape=( bian_225,  bian_225, 3), dtype=np.uint8)
        img_padded_225[padding_h_225:padding_h_225+h, padding_w_225:padding_w_225+w, :] = img
        M_225 = cv.getRotationMatrix2D(center_225, 225, 1)
        dst7 = cv.warpAffine(img_padded_225, M_225, ( bian_225,  bian_225))
        rote_225 = dst7
        ############### 旋转225 ######################
        mat_270 = cv.getRotationMatrix2D((w//2, h//2), -270, 1)
        Rot_270 = cv.invertAffineTransform(mat_270)
        vertex_270 =np.dot(Rot_270,np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]])) # 求得变换后的坐标
        bian_270=max(int((max(vertex_270[0])-min(vertex_270[0]))),int((max(vertex_270[1])-min(vertex_270[1]))))
        padding_h_270 = int((bian_270-h) // 2)
        padding_w_270 = int(( bian_270-w) // 2)
        center_270 = ( bian_270 // 2,  bian_270 // 2)
        img_padded_270 = np.zeros(shape=( bian_270,  bian_270, 3), dtype=np.uint8)
        img_padded_270[padding_h_270:padding_h_270+h, padding_w_270:padding_w_270+w, :] = img
        M_270 = cv.getRotationMatrix2D(center_270, 270, 1)
        dst8 = cv.warpAffine(img_padded_270, M_270, ( bian_270,  bian_270))
        rote_270 = dst8
        ############### 旋转225 ######################
        mat_315 = cv.getRotationMatrix2D((w//2, h//2), -315, 1)
        Rot_315 = cv.invertAffineTransform(mat_315)
        vertex_315 =np.dot(Rot_315,np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]])) # 求得变换后的坐标
        bian_315=max(int((max(vertex_315[0])-min(vertex_315[0]))),int((max(vertex_315[1])-min(vertex_315[1]))))
        padding_h_315 = int((bian_315-h) // 2)
        padding_w_315 = int(( bian_315-w) // 2)
        center_315 = ( bian_315 // 2,  bian_315 // 2)
        img_padded_315 = np.zeros(shape=( bian_315,  bian_315, 3), dtype=np.uint8)
        img_padded_315[padding_h_315:padding_h_315+h, padding_w_315:padding_w_315+w, :] = img
        M_315 = cv.getRotationMatrix2D(center_315, 315, 1)
        dst9 = cv.warpAffine(img_padded_315, M_315, ( bian_315,  bian_315))
        rote_315 = dst9

#################################################################################

 ################ 高斯模糊 ######################################################
        dst10 = cv.GaussianBlur(img, (5, 5), 0)
        dst11 = cv.GaussianBlur(flip_x, (5, 5), 0)
        dst12 = cv.GaussianBlur(flip_y, (5, 5), 0)
        dst13 = cv.GaussianBlur(flip_x_y, (5, 5), 0)
        dst14 = cv.GaussianBlur(rote_45, (5, 5), 0)
        dst15 = cv.GaussianBlur(rote_90, (5, 5), 0)
        dst16 = cv.GaussianBlur(rote_135, (5, 5), 0)
        dst17 = cv.GaussianBlur(rote_225, (5, 5), 0)
        dst18 = cv.GaussianBlur(rote_270, (5, 5), 0)
        dst19 = cv.GaussianBlur(rote_315, (5, 5), 0)

 ################ 曝光改变 ######################################################
        # contrast
        reduce = 0.5
        increase = 1.4
        # brightness
        g = 10
        h, w, ch = img.shape

        add = np.zeros([h, w, ch], img.dtype)
        add_45 = np.zeros([bian_45, bian_45, 3], rote_45.dtype)
        add_90 = np.zeros([bian_90, bian_90, 3], rote_90.dtype)
        add_135 = np.zeros([bian_135, bian_135, 3], rote_135.dtype)
        add_225 = np.zeros([bian_225, bian_225, 3], rote_225.dtype)
        add_270 = np.zeros([bian_270, bian_270, 3], rote_270.dtype)
        add_315 = np.zeros([bian_315, bian_315, 3], rote_315.dtype)

        dst20 = cv.addWeighted(img, reduce, add, 1-reduce, g)
        dst21 = cv.addWeighted(flip_x, reduce, add, 1 - reduce, g)
        dst22 = cv.addWeighted(flip_y, reduce, add, 1 - reduce, g)
        dst23 = cv.addWeighted(flip_x_y, reduce, add, 1 - reduce, g)            
        dst24 = cv.addWeighted(rote_45, reduce, add_45, 1 - reduce, g)
        dst25 = cv.addWeighted(rote_90, reduce, add_90, 1 - reduce, g)
        dst26 = cv.addWeighted(rote_135, reduce, add_135, 1 - reduce, g)
        dst27 = cv.addWeighted(rote_225, reduce, add_225, 1 - reduce, g)
        dst28 = cv.addWeighted(rote_270, reduce, add_270, 1 - reduce, g)
        dst29 = cv.addWeighted(rote_315, reduce, add_315, 1 - reduce, g)
        dst30 = cv.addWeighted(img, increase, add, 1-increase, g)
        dst31 = cv.addWeighted(flip_x, increase, add, 1 - increase, g)            
        dst32 = cv.addWeighted(flip_y, increase, add, 1 - increase, g)
        dst33 = cv.addWeighted(flip_x_y, increase, add, 1 - increase, g)
        dst34 = cv.addWeighted(rote_45, increase, add_45, 1 - increase, g)
        dst35 = cv.addWeighted(rote_90, increase, add_90, 1 - increase, g)
        dst36 = cv.addWeighted(rote_135, increase, add_135, 1 - increase, g)
        dst37 = cv.addWeighted(rote_225, increase, add_225, 1 - increase, g)
        dst38 = cv.addWeighted(rote_270, increase, add_270, 1 - increase, g)
        dst39 = cv.addWeighted(rote_315, increase, add_315, 1 - increase, g)

 ################ 增加椒盐噪声 ######################################################
        percentage = 0.005
        dst40 = img
        dst41 = flip_x
        dst42 = flip_y
        dst43 = flip_x_y
        dst44 = rote_45
        dst45 = rote_90
        dst46 = rote_135
        dst47 = rote_225
        dst48 = rote_270
        dst49 = rote_315
        num = int(percentage * img.shape[0] * img.shape[1])
        for i in range(num):
             rand_x = random.randint(0, img.shape[0] - 1)
             rand_y = random.randint(0, img.shape[1] - 1)
             if random.randint(0, 1) == 0:
                 dst40[rand_x, rand_y] = 0
                 dst41[rand_x, rand_y] = 0
                 dst42[rand_x, rand_y] = 0
                 dst43[rand_x, rand_y] = 0
                 dst44[rand_x, rand_y] = 0
                 dst45[rand_x, rand_y] = 0
                 dst46[rand_x, rand_y] = 0
                 dst47[rand_x, rand_y] = 0
                 dst48[rand_x, rand_y] = 0
                 dst49[rand_x, rand_y] = 0
             else:
                 dst40[rand_x, rand_y] = 255
                 dst41[rand_x, rand_y] = 255
                 dst42[rand_x, rand_y] = 255
                 dst43[rand_x, rand_y] = 255
                 dst44[rand_x, rand_y] = 255
                 dst45[rand_x, rand_y] = 255
                 dst46[rand_x, rand_y] = 255
                 dst47[rand_x, rand_y] = 255
                 dst48[rand_x, rand_y] = 255
                 dst49[rand_x, rand_y] = 255

############### 色彩抖动 ###########################################
        imge= Image.open(str(image_path)+'/'+str(image_id)+str(image_type))
        flip_x_x= ImageOps.flip(imge)  
        flip_y_y= imge.transpose(Image.FLIP_LEFT_RIGHT) 
        flip_x_y_x_y= flip_x_x.transpose(Image.FLIP_LEFT_RIGHT)  
        random_factor1 = np.random.randint(0, 31) / 10.  # 随机因子
        dst50 = ImageEnhance.Color(imge).enhance(random_factor1)  # 调整图像的饱和度
        dst51 = ImageEnhance.Color(flip_x_x).enhance(random_factor1)
        dst52 = ImageEnhance.Color(flip_y_y).enhance(random_factor1)
        dst53 = ImageEnhance.Color(flip_x_y_x_y).enhance(random_factor1)   

        random_factor2 = np.random.randint(10, 21) / 10.  # 随机因1子
        dst54 = ImageEnhance.Contrast(imge).enhance(random_factor2)  # 调整图像对比度
        dst55 = ImageEnhance.Contrast(flip_x_x).enhance(random_factor2)  
        dst56 = ImageEnhance.Contrast(flip_y_y).enhance(random_factor2) 
        dst57 = ImageEnhance.Contrast(flip_x_y_x_y).enhance(random_factor2) 

        random_factor3 = np.random.randint(0, 31) / 10.  # 随机因子
        dst58 = ImageEnhance.Sharpness(imge).enhance(random_factor3) # 调整图像锐度
        dst59 = ImageEnhance.Sharpness(flip_x_x).enhance(random_factor3)
        dst60 = ImageEnhance.Sharpness(flip_y_y).enhance(random_factor3)
        dst61 = ImageEnhance.Sharpness(flip_x_y_x_y).enhance(random_factor3)

        dst62 = ImageEnhance.Sharpness(dst50).enhance(random_factor3)
        dst63 = ImageEnhance.Contrast(dst50).enhance(random_factor2)
        dst64 = ImageEnhance.Sharpness(dst63).enhance(random_factor3)
                
######################## 生成标签文件 ##################################3  
       
    #def json_generation(self):
        image_names =[]
        
        for case in cases:
            if case==1:
               cv.imwrite(savepath+'/'+str(image_id)+'_flip_x'+'.jpg', flip_x)
               image_names.append(str(image_id)+'_flip_x')
            elif case==2:               
               cv.imwrite(savepath+'/'+str(image_id)+'_flip_y'+'.jpg', flip_y)
               image_names.append(str(image_id)+'_flip_y')
            elif case==3:               
               cv.imwrite(savepath+'/'+str(image_id)+'_flip_x_y'+'.jpg', flip_x_y)
               image_names.append( str(image_id)+'_flip_x_y')
            elif case==4:              
               cv.imwrite(savepath+'/'+str(image_id)+'_rote_45'+'.jpg', cv.resize(rote_45,(w,h)))
               image_names.append(str(image_id)+'_rote_45')
            elif case==5:  
               cv.imwrite(savepath+'/'+str(image_id)+'_rote_90'+'.jpg', cv.resize(rote_90,(w,h)))
               image_names.append(str(image_id)+'_rote_90') 
            elif case==6:  
               cv.imwrite(savepath+'/'+str(image_id)+'_rote_135'+'.jpg', cv.resize(rote_135,(w,h)))
               image_names.append(str(image_id)+'_rote_135') 
            elif case==7:  
               cv.imwrite(savepath+'/'+str(image_id)+'_rote_225'+'.jpg', cv.resize(rote_225,(w,h)))
               image_names.append(str(image_id)+'_rote_225')
            elif case==8:  
               cv.imwrite(savepath+'/'+str(image_id)+'_rote_270'+'.jpg', cv.resize(rote_270,(w,h)))
               image_names.append(str(image_id)+'_rote_270')
            elif case==9:  
               cv.imwrite(savepath+'/'+str(image_id)+'_rote_315'+'.jpg', cv.resize(rote_315,(w,h)))
               image_names.append(str(image_id)+'_rote_315')
            elif case==10: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_Gaussian'+'.jpg', dst10)  
                 image_names.append(str(image_id)+'_Gaussian')
            elif case==11: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x'+'_Gaussian'+'.jpg', dst11)
                 image_names.append(str(image_id)+'_flip_x'+'_Gaussian')
            elif case==12:
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_y'+'_Gaussian'+'.jpg', dst12)
                 image_names.append(str(image_id)+'_flip_y' + '_Gaussian')
            elif case==13: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x_y'+'_Gaussian'+'.jpg', dst13)
                 image_names.append(str(image_id)+'_flip_x_y'+'_Gaussian')
            elif case==14: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_45'+'_Gaussian'+'.jpg', cv.resize(dst14,(w,h)))
                 image_names.append(str(image_id)+'_rote_45'+'_Gaussian')
            elif case==15: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_90'+'_Gaussian'+'.jpg', cv.resize(dst15,(w,h)))
                 image_names.append(str(image_id)+'_rote_90' + '_Gaussian')
            elif case==16: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_135'+'_Gaussian'+'.jpg', cv.resize(dst16,(w,h)))
                 image_names.append(str(image_id)+'_rote_135'+'_Gaussian')
            elif case==17: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_225'+'_Gaussian'+'.jpg', cv.resize(dst17,(w,h)))
                 image_names.append(str(image_id)+'_rote_225'+'_Gaussian')
            elif case==18: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_270'+'_Gaussian'+'.jpg', cv.resize(dst18,(w,h)))
                 image_names.append(str(image_id)+'_rote_270' + '_Gaussian')
            elif case==19: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_315'+'_Gaussian'+'.jpg', cv.resize(dst19,(w,h)))
                 image_names.append(str(image_id)+'_rote_315'+'_Gaussian')
            elif case==20: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_ReduceEp'+'.jpg', dst20)
                 image_names.append(str(image_id)+'_ReduceEp')
            elif case==21: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x'+'_ReduceEp'+'.jpg', dst21)
                 image_names.append(str(image_id)+'_flip_x'+'_ReduceEp')
            elif case==22: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_y'+'_ReduceEp'+'.jpg', dst22)
                 image_names.append(str(image_id)+'_flip_y'+'_ReduceEp')
            elif case==23: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x_y'+'_ReduceEp'+'.jpg', dst23)
                 image_names.append(str(image_id)+'_flip_x_y'+'_ReduceEp')
            elif case==24: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_45'+'_ReduceEp'+'.jpg', cv.resize(dst24,(w,h)))
                 image_names.append(str(image_id)+'_rote_45'+'_ReduceEp')
            elif case==25:
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_90'+'_ReduceEp'+'.jpg', cv.resize(dst25,(w,h)))
                 image_names.append(str(image_id)+'_rote_90'+'_ReduceEp')
            elif case==26: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_135'+'_ReduceEp'+'.jpg', cv.resize(dst26,(w,h)))
                 image_names.append(str(image_id)+'_rote_135'+'_ReduceEp')
            elif case==27: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_225'+'_ReduceEp'+'.jpg', cv.resize(dst27,(w,h)))
                 image_names.append(str(image_id)+'_rote_225'+'_ReduceEp')
            elif case==28:
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_270'+'_ReduceEp'+'.jpg', cv.resize(dst28,(w,h)))
                 image_names.append(str(image_id)+'_rote_270'+'_ReduceEp')
            elif case==29: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_315'+'_ReduceEp'+'.jpg', cv.resize(dst29,(w,h)))
                 image_names.append(str(image_id)+'_rote_315'+'_ReduceEp')
            elif case==30: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_IncreaseEp'+'.jpg', dst30)
                 image_names.append(str(image_id)+'_IncreaseEp')
            elif case==31:
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x'+'_IncreaseEp'+'.jpg', dst31)
                 image_names.append(str(image_id)+'_flip_x'+'_IncreaseEp')
            elif case==32: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_y'+'_IncreaseEp'+'.jpg', dst32)
                 image_names.append(str(image_id)+'_flip_y'+'_IncreaseEp')
            elif case==33: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x_y'+'_IncreaseEp'+'.jpg', dst33)
                 image_names.append(str(image_id)+'_flip_x_y'+'_IncreaseEp')
            elif case==34: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_45'+'_IncreaseEp'+'.jpg', cv.resize(dst34,(w,h)))
                 image_names.append(str(image_id)+'_rote_45'+'_IncreaseEp')
            elif case==35: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_90'+'_IncreaseEp'+'.jpg', cv.resize(dst35,(w,h)))
                 image_names.append(str(image_id)+'_rote_90'+'_IncreaseEp')
            elif case==36: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_135'+'_IncreaseEp'+'.jpg', cv.resize(dst36,(w,h)))
                 image_names.append(str(image_id)+'_rote_135'+'_IncreaseEp')
            elif case==37:
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_225'+'_IncreaseEp'+'.jpg', cv.resize(dst37,(w,h)))
                 image_names.append(str(image_id)+'_rote_225'+'_IncreaseEp')
            elif case==38: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_270'+'_IncreaseEp'+'.jpg', cv.resize(dst38,(w,h)))
                 image_names.append(str(image_id)+'_rote_270'+'_IncreaseEp')
            elif case==39: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_315'+'_IncreaseEp'+'.jpg', cv.resize(dst39,(w,h)))
                 image_names.append(str(image_id)+'_rote_315'+'_IncreaseEp')
            elif case==40: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_Salt'+'.jpg', dst40)
                 image_names.append(str(image_id)+'_Salt')
            elif case==41: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x'+'_Salt'+'.jpg', dst41)
                 image_names.append(str(image_id)+'_flip_x' + '_Salt')
            elif case==42:
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_y'+'_Salt'+'.jpg', dst42)
                 image_names.append(str(image_id)+'_flip_y' + '_Salt')
            elif case==43: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_flip_x_y'+'_Salt'+'.jpg', dst43)
                 image_names.append(str(image_id)+'_flip_x_y' + '_Salt')
            elif case==44: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_45'+'_Salt'+'.jpg', cv.resize(dst44,(w,h)))
                 image_names.append(str(image_id)+'_rote_45'+'_Salt')
            elif case==45:
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_90'+'_Salt'+'.jpg', cv.resize(dst45,(w,h)))
                 image_names.append(str(image_id)+'_rote_90'+'_Salt')
            elif case==46: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_135'+'_Salt'+'.jpg', cv.resize(dst46,(w,h)))
                 image_names.append(str(image_id)+'_rote_135'+'_Salt')
            elif case==47: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_225'+'_Salt'+'.jpg', cv.resize(dst47,(w,h)))
                 image_names.append(str(image_id)+'_rote_225'+'_Salt')
            elif case==48:
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_270'+'_Salt'+'.jpg', cv.resize(dst48,(w,h)))
                 image_names.append(str(image_id)+'_rote_270'+'_Salt')
            elif case==49: 
                 cv.imwrite(savepath+'/'+str(image_id)+'_rote_315'+'_Salt'+'.jpg', cv.resize(dst49,(w,h)))
                 image_names.append(str(image_id)+'_rote_315'+'_Salt')
            elif case==50: 
                 dst50.save(savepath+'/'+str(image_id)+'_Color'+'.jpg')
                 image_names.append(str(image_id)+'_Color')
            elif case==51:
                 dst51.save(savepath+'/'+str(image_id)+'_flip_x'+'_Color'+'.jpg')
                 image_names.append(str(image_id)+'_flip_x' + '_Color')
            elif case==52: 
                 dst52.save(savepath+'/'+str(image_id)+'_flip_y'+'_Color'+'.jpg')
                 image_names.append(str(image_id)+'_flip_y' + '_Color')
            elif case==53: 
                 dst53.save(savepath+'/'+str(image_id)+'_flip_x_y'+'_Color'+'.jpg')
                 image_names.append(str(image_id)+'_flip_x_y' + '_Color')
            elif case==54: 
                 dst54.save(savepath+'/'+str(image_id)+'_Contrast'+'.jpg')
                 image_names.append(str(image_id)+'_Contrast')
            elif case==55: 
                 dst55.save(savepath+'/'+str(image_id)+'_flip_x'+'_Contrast'+'.jpg')
                 image_names.append(str(image_id)+'_flip_x' + '_Contrast')
            elif case==56: 
                 dst56.save(savepath+'/'+str(image_id)+'_flip_y'+'_Contrast'+'.jpg')
                 image_names.append(str(image_id)+'_flip_y' + '_Contrast')
            elif case==57:
                 dst57.save(savepath+'/'+str(image_id)+'_flip_x_y'+'_Contrast'+'.jpg')
                 image_names.append(str(image_id)+'_flip_x_y' + '_Contrast')
            elif case==58: 
                 dst58.save(savepath+'/'+str(image_id)+'_Sharpness'+'.jpg')
                 image_names.append(str(image_id)+'_Sharpness')
            elif case==59: 
                 dst59.save(savepath+'/'+str(image_id)+'_flip_x'+'_Sharpness'+'.jpg')
                 image_names.append(str(image_id)+'_flip_x' + '_Sharpness')
            elif case==60:
                 dst60.save(savepath+'/'+str(image_id)+'_flip_y'+'_Sharpness'+'.jpg')
                 image_names.append(str(image_id)+'_flip_y' + '_Sharpness')
            elif case==61: 
                 dst61.save(savepath+'/'+str(image_id)+'_flip_x_y'+'_Sharpness'+'.jpg')
                 image_names.append(str(image_id)+'_flip_x_y' + '_Sharpness')
            elif case==62: 
                 dst62.save(savepath+'/'+str(image_id)+'_Color'+'_Sharpness'+'.jpg')
                 image_names.append(str(image_id)+'_Color'+'_Sharpness')
            elif case==63:
                 dst63.save(savepath+'/'+str(image_id)+'_Color'+'_Contrast'+'.jpg')
                 image_names.append(str(image_id)+'_Color'+'_Contrast')
            elif case==64:
                 dst64.save(savepath+'/'+str(image_id)+'_Color'+'_Contrast'+'_Sharpness'+'.jpg')
                 image_names.append(str(image_id)+'_Color'+'_Contrast'+'_Sharpness')

        for image_name in image_names:
            
            with open(str(savepath)+'/'+image_name+".jpg", "rb") as b64:
                base64_data_original = str(base64.b64encode(b64.read()))
                base64_data = base64_data_original
            with open(str(label_path)+'/'+str(image_id)+".json", 'r')as js:
                json_data = json.load(js)
                #img = utils.img_b64_to_arr(json_data['imageData'])
                #height, width = img.shape[:2]
                shapes = json_data['shapes']
                for shape in shapes:
                    points = shape['points']
                    for point in points:
                        match_pattern2 = re.compile(r'(.*)_x(.*)')
                        match_pattern3 = re.compile(r'(.*)_y(.*)')
                        match_pattern4 = re.compile(r'(.*)_x_y(.*)')
                        match_pattern5 = re.compile(r'(.*)_rote_45(.*)')
                        match_pattern6 = re.compile(r'(.*)_rote_90(.*)')
                        match_pattern7 = re.compile(r'(.*)_rote_135(.*)')
                        match_pattern8 = re.compile(r'(.*)_rote_225(.*)')
                        match_pattern9 = re.compile(r'(.*)_rote_270(.*)')
                        match_pattern10 = re.compile(r'(.*)_rote_315(.*)')
                        if match_pattern4.match(image_name):
                            point[0] = w - point[0]
                            point[1] = h - point[1]
                        elif match_pattern3.match(image_name):
                            point[0] = w - point[0]
                            point[1] = point[1]
                        elif match_pattern2.match(image_name):
                            point[0] = point[0]
                            point[1] = h - point[1]
                        elif match_pattern5.match(image_name):
                            pointsss=np.dot(M_45,np.array([[point[0]+(bian_45-w)//2],[point[1]+(bian_45-h)//2],[1]])) # 求得变换后的坐标
                            point[0]=int(pointsss[0,0]*(w/bian_45))
                            point[1]=int(pointsss[1,0]*(h/bian_45))

                        elif match_pattern6.match(image_name):
                            pointsss=np.dot(M_90,np.array([[point[0]+(bian_90-w)//2],[point[1]+(bian_90-h)//2],[1]])) # 求得变换后的坐标
                            point[0]=int(pointsss[0,0]*(w/bian_90))
                            point[1]=int(pointsss[1,0]*(h/bian_90))
                        elif match_pattern7.match(image_name):
                            pointsss=np.dot(M_135,np.array([[point[0]+(bian_135-w)//2],[point[1]+(bian_135-h)//2],[1]])) # 求得变换后的坐标
                            point[0]=int(pointsss[0,0]*(w/bian_135))
                            point[1]=int(pointsss[1,0]*(h/bian_135))
                        elif match_pattern8.match(image_name):
                            pointsss=np.dot(M_225,np.array([[point[0]+(bian_225-w)//2],[point[1]+(bian_225-h)//2],[1]])) # 求得变换后的坐标
                            point[0]=int(pointsss[0,0]*(w/bian_225))
                            point[1]=int(pointsss[1,0]*(h/bian_225))
                        elif match_pattern9.match(image_name):
                            pointsss=np.dot(M_270,np.array([[point[0]+(bian_270-w)//2],[point[1]+(bian_270-h)//2],[1]])) # 求得变换后的坐标
                            point[0]=int(pointsss[0,0]*(w/bian_270))
                            point[1]=int(pointsss[1,0]*(h/bian_270))
                        elif match_pattern10.match(image_name):
                            pointsss=np.dot(M_315,np.array([[point[0]+(bian_315-w)//2],[point[1]+(bian_315-h)//2],[1]])) # 求得变换后的坐标
                            point[0]=int(pointsss[0,0]*(w/bian_315))
                            point[1]=int(pointsss[1,0]*(h/bian_315))
                        else:
                            point[0] = point[0]
                            point[1] = point[1]
                json_data['imagePath'] = str(n).zfill(6)+".jpg"
                
                json_data['imageData'] = base64_data[2:] ## 去掉前两个字符
                json.dump(json_data, open(save_label_path+'/'+str(n).zfill(6)+".json", 'w'), indent=2)
                
def add(num_agu,imagpath_visible,imagpath_lwir,savepath_lwir,savepath_visible,imagetype,labelpath,save_img_path_lwir,save_img_path_visible,save_label_path_visible,save_label_path_lwir):
    path_list=os.listdir(imagpath_visible)
    path_list.sort(key=lambda x: int(x.split('.')[0]))
    num_file=len(path_list)
    n=25000
    nv=n
    nl=n
    for i in range(num_file):
        imagid=str(path_list[i].split('.')[0])
        print(imagid)
        cases=random.sample(range(1, 65), num_agu)
        print('cases=====',cases) 
        dataAugmentObject = DataAugment(num_agu,imagpath_visible,imagid,savepath_visible,imagetype,labelpath,cases,save_label_path_visible,n)
        dataAugmentObject = DataAugment(num_agu,imagpath_lwir,imagid,savepath_lwir,imagetype,labelpath,cases,save_label_path_lwir,n)
        n=n+1
    path_list_json=glob.glob(os.path.join(savepath_visible, '*.jpg'))
    path_list_json.sort(key=lambda x: int(x.replace(savepath_visible+"/","").split('_')[0]))
    num_json=len(path_list_json)
    for i in range(num_json):
        path=path_list_json[i]
        print(path)
        imgid=str(path_list[i].replace(savepath_visible+"/","").split('_')[0])
        print('id:',imgid)
        img=cv.imread(path)
        cv.imwrite(save_img_path_visible+str(nv).zfill(6)+'.jpg',img)
        nv=nv+1
    path_list_json=glob.glob(os.path.join(savepath_lwir, '*.jpg'))
    path_list_json.sort(key=lambda x: int(x.replace(savepath_lwir+"/","").split('_')[0]))
    num_json=len(path_list_json)
    for i in range(num_json):
        path=path_list_json[i]
        print(path)
        imgid=str(path_list[i].replace(savepath_lwir+"/","").split('_')[0])
        print('id:',imgid)
        img=cv.imread(path)
        cv.imwrite(save_img_path_lwir+str(nl).zfill(6)+'.jpg',img)
        nl=nl+1
        

#if __name__ == "__main__":
imagpath_visible='visible_ori'
imagpath_lwir='lwir_ori'
savepath_lwir='tmp_lwir'
savepath_visible='tmp_visible'
#imagid='000001'
imagetype='.jpg'
num_agu=1
labelpath='./annotations'
save_label_path_visible='./label_json_visible/'
save_label_path_lwir='./label_json_lwir/'
save_img_path_lwir='./lwir/'
save_img_path_visible='./visible/'
#dataAugmentObject = DataAugment(num_agu,imagpath,imagid,savepath,imagetype,labelpath)
add(num_agu,imagpath_visible,imagpath_lwir,savepath_lwir,savepath_visible,imagetype,labelpath,save_img_path_lwir,save_img_path_visible,save_label_path_visible,save_label_path_lwir)


