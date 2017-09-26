# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:01:29 2017

@author: Shaoshen Wang
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2


def align(source,template,X=0,Y=0,RANGE = 15,is_pyramid = False):#SAD
    s_height,s_width = source.shape[0],source.shape[1]
    t_height,t_width = template.shape[0],template.shape[1]
    
    print(source.shape,template.shape,(X,Y),RANGE)
    INF = float('inf')
    D_min = INF
    pos_min = (0,0)
    
    for s_y in range(Y-RANGE,Y+RANGE):
        for s_x in range(X-RANGE,X+RANGE):
            
            cut = source[s_y:s_y+t_height,s_x:s_x+t_width]
            D = sum(sum(abs(template/255-cut/255)))  #format :0-255
            if D < D_min:
                pos_min = (s_x,s_y)
                D_min = D
    if not is_pyramid:            
        return (pos_min[0]-X,pos_min[1]-Y)
    else:
        return (pos_min[0],pos_min[1])
    
def pyramid(source,template,search_range,X=0,Y=0,x_c=0,y_c=0,RANGE=5):
    #searching center in orinial image:(X,Y) searching center in level3 of pyrimid:(x_c,y_c)
    #search_range: search range in different level of pyrimid
    level = 3 # total level of pyrimid
    pos = (x_c,y_c)
    #search_range = (3,3,10)
    sources = [0]*level
    temps = [0]*level
    sources[0] = source #store
    temps[0] = template
    
    for l in range(1,level):
        sources[l] = cv2.pyrDown(sources[l-1])
        temps[l] = cv2.pyrDown(temps[l-1])
        #cv2.imshow('image',imgs[l])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    for i in range(level-1,-1,-1):
        #print(sources[i].shape,temps[i].shape,pos)
        pos = align(sources[i],temps[i],pos[0],pos[1],RANGE=search_range[i],is_pyramid=True)       
        pos = (pos[0]*2,(pos[1]*2))
        
    shift = (pos[0]/2-X,(pos[1]/2-Y))#remove abundent multiplication   
    return shift
    
def translate(image, x, y):
    # shifting
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted

def calcAndDrawHist(image, color):    
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)    
    histImg = np.zeros([256,256,3], np.uint8)    
    hpt = int(0.9* 256);    
        
    for h in range(256):    
        intensity = int(hist[h]*hpt/maxVal)    
        cv2.line(histImg,(h,256), (h,256-intensity), color)    
            
    return histImg

def showHistogram(img):
    b, g, r = cv2.split(img)    
    histImgB = calcAndDrawHist(b, [255, 0, 0])    
    histImgG = calcAndDrawHist(g, [0, 255, 0])    
    histImgR = calcAndDrawHist(r, [0, 0, 255])    
    cv2.imshow("histImgB", histImgB)    
    cv2.imshow("histImgG", histImgG)    
    cv2.imshow("histImgR", histImgR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def removeEdge(img):
    e = []#l,r,t,b
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height,width = img.shape[0],img.shape[1]
    length = int(0.08*(height+width)/2)
    #print(length)
    img = cv2.GaussianBlur(img,(3,3),0)  
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)  
    
    count=[0]
    for w in range(length):
        p=edges[height//2,w]
        if p==255:
            count.append(w)
    e.append(count[-1])
    count=[width-1]       
    for w in range(width-1,width-length-1,-1):
        
        p=edges[height//2,w]
        if p==255:
            count.append(w) 
    e.append(count[-1])
    count=[0]
    
    for h in range(length):
        p=edges[h,width//2]
        if p==255:
            count.append(h)
            
    e.append(count[-1])
    count=[height-1]
    for h in range(height-1,height-length-1,-1):
        p=edges[h,width//2]
        #print(p)
        if p==255:
            count.append(h)
    e.append(count[-1])
    print(e)
    cv2.imshow('Canny', edges)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return e
###############tasks##########

def task1(img_name,r_shift=15,R=200,X=50,Y=50): #RANGE:template range  
    img = cv2.imread(img_name,0)
    #print(type(img))
    height,width = img.shape[0],img.shape[1]

    each_height = int(height/3)

    blue = img[0:each_height,:]
    red = img[each_height:each_height*2,:]
    green = img[each_height*2:each_height*3,:]
    
    '''                        
    cv2.imshow('image',green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    red_shift = align(blue,red[Y:Y+R,X:X+R],X,Y,RANGE=r_shift)
    green_shift= align(blue,green[Y:Y+R,X:X+R],X,Y,RANGE=r_shift)
    print(red_shift)
    print(green_shift)
    
        
    red = translate(red, red_shift[0], red_shift[1])
    green = translate(green, green_shift[0], green_shift[1])
    
    merged = cv2.merge([blue,red,green]) 
    cv2.imwrite(img_name[:-4]+"_"+".jpg", merged)
    cv2.imshow('image',merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task2(img_name,X=10,Y=10,RANGE=200,x_c=10,y_c=10,search_range=(3,3,10)): #RANGE:template range #(x_c,y_c) smallest image
    #RANGE:template scale
    #search_range: search range in pyrimid
    img = cv2.imread(img_name,0)
    height,width = img.shape[0],img.shape[1]
    #print(height,width)
    each_height = int(height/3)

    blue = img[0:each_height,:]
    red = img[each_height:each_height*2,:]
    green = img[each_height*2:each_height*3,:]
    
    red_pos = pyramid(blue,red[Y:Y+RANGE,X:X+RANGE],search_range,X,Y,x_c,y_c)
    green_pos = pyramid(blue,green[Y:Y+RANGE,X:X+RANGE],search_range,X,Y,x_c,y_c)
    print(red_pos,green_pos)
    red = translate(red, red_pos[0], red_pos[1])
    green = translate(green, green_pos[0], green_pos[1])
    merged = cv2.merge([blue,red,green])
    
    cv2.imwrite(img_name[:-4]+"_"+".jpg", merged)
    cv2.imshow('image',merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def task3(img_name,edge_ratio=0.05):
    img = cv2.imread(img_name,1)
    
    height,width = img.shape[0],img.shape[1]
    print(img.shape)
    
    lut = np.zeros(256, dtype = img.dtype )#Create new lookup table  
  
    hist,bins = np.histogram(img.flatten(),256,[0,256])   
    cdf = hist.cumsum() #cumulate histogram 
    cdf_m = np.ma.masked_equal(cdf,0) #delete 0 in histogram 
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#lut[i] = int(255.0 *p[i])
    cdf = np.ma.filled(cdf_m,0).astype('uint8') #padding 0  
    
    #result2 = cdf[img]  
    result = cv2.LUT(img, cdf)  
    edge = removeEdge(result)    #l,r,t,b 
    
    #print(edge)
    #result_withoutedge = result[int(height*edge_ratio):int(-height*edge_ratio),int(width*edge_ratio):int(-width*edge_ratio)]
    result_withoutedge = result[edge[2]:edge[3],edge[0]:edge[1]]
    cv2.imwrite(img_name[:-4]+"_enhanced"+".jpg", result_withoutedge)
    #showHistogram(result)
       
    
    cv2.imshow("OpenCVLUT", result_withoutedge) 
    #cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    small="./DataSamples/s1.jpg"
    small = "./test_samples/test_sample1.jpg"
    big = "./test_samples/test_sample2.jpg"
    #small
    task1(img_name=small,R=200,r_shift=20,X=100,Y=100)
    #task2(img_name=small,search_range=(2,2,8))
    #task3("./test_samples/test_sample1_.jpg")
    
    #big
    #task1(img_name=big,X=1800,Y=1000,R=300,r_shift = 60)
    task2(big,X=50,Y=50,RANGE =1500,x_c=30,y_c=30,search_range = (3,3,20))
    
    #task2(big,X=50,Y=50,RANGE = 1200,x_c=30,y_c=30,search_range = (3,3,20))
    #task3("./DataSamples/b2_r.jpg")
    
    #huge
    #task2("./DataSamples/bb1.tif",X=50,Y=50,R = 1500,x_c=20,y_c=20,search_range = (3,3,10))
    #b1((-5, 5)
    #(-7, 59))
    #b2(-8, 23)
    #(-17, 56)
    #white balance