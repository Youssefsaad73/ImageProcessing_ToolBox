from __future__ import print_function
import os
from tkinter import S
from turtle import color
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog,QLabel,QDialog,QMessageBox
from PyQt5 import uic, QtGui, QtCore
import sys
from cv2 import resize
import cv2
import matplotlib
from matplotlib.axis import YAxis
from fileinput import filename
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image as ImagePil, ImageOps
from PIL.ImageQt import ImageQt
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QFileDialog
import numpy as np
from scipy import misc
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as npy
import pyqtgraph as pg
import math  
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import random
import matplotlib.image as img
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog,QLabel,QDialog,QMessageBox,QRubberBand
from matplotlib.widgets  import RectangleSelector
from skimage.data import shepp_logan_phantom 
from skimage.transform import radon , rescale , iradon
import scipy

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi(r"ImageTask3(final).ui", self)
        self.glayout= pg.GraphicsLayout()
        self.glayout=pg.GraphicsLayout()
        self.vb=self.glayout.addViewBox()
        self.browse_pushButton.clicked.connect(self.BrowseImage) ###connections###
        self.BasicInfo_tableWidget.setColumnWidth(0,420)
        self.BasicInfo_tableWidget.setColumnWidth(1,420)
        self.Zoom_pushButton.clicked.connect(self.Zoom_NearestNeighbourInterpolation)
        self.Zoom_pushButton.clicked.connect(self.PlotLinearInterpolation)
        #self.Zoom_pushButton.clicked.connect(self.Zoom_LinearInterpolation)
        self.nnH_lineEdit.textEdited.connect(self.SetScaleFactor)
        self.Apply_pushButton.clicked.connect(self.PlotConstructedImage)
        self.nearestneighbour_pushButton.clicked.connect(self.rotate)
        self.Shear_pushButton.clicked.connect(self.naive_shearing)
        self.Direction_lineEdit.hide()
        self.lineEdit.textEdited.connect(self.SetInformationDirection)
        self.ShowOriginalHistogram_pushButton.clicked.connect(self.Historgram1)
        self.sharpening_pushButton.clicked.connect(self.BoxKernel)
        self.addingsaltandpepper_pushButton.clicked.connect(self.SandP)
        self.removingsaltandpepper_pushButton.clicked.connect(self.median_filter)
        self.pushButton.clicked.connect(self.fft)
        self.pushButton_2.clicked.connect(self.FourierFilter)
        self.removepattern_pushButton.clicked.connect(self.removepattern)
        self.pushButton_3.clicked.connect(self.fourier2)
        #self.histogram1_graphicsView.canvas.axes.set_facecolor((183.0,134.0,32.0))
        self.phantom_pushButton.clicked.connect(self.phantom_gen)
        self.addnoise_pushButton.clicked.connect(self.noisy)
        self.selectroi_pushButton.clicked.connect(self.selectroi)
        self.Load_phantom_pushButton.clicked.connect(self.B_P_function)
        self.Load_phantom_pushButton.clicked.connect(self.B_P_functionn)
        



        test=np.array([[1,3,7],
                            [3,5,7],
                            [5,7,9]])
        #test=self.resizeLayer(test)
        #test=self.PlotLinearInterpolation(test)
        print('final',test)
        
        #cv2.destroyAllWindows()
        ###
        

        self.show()

        ############Browse############
    def BrowseImage(self):
        self.BasicInfo_tableWidget.clearContents()
        global file_name   ###opening file###
        file_name=QFileDialog.getOpenFileName(self, "Browse Image", "../", "*.dcm;;" "*.bmp;;" "*.jpeg;;" "*.png;;" )
        global file_path
        file_path =file_name[0]
        global Pil_img
        ##

        #cv2.waitKey(0)
       
        ####check if the file is dicom or jpeg or bmp###
        if file_name[0].endswith(".dcm"):
            Pil_img=self.setDicomImage(file_path)  
            Pixmap_img=Pil_img.toqpixmap()
            Pixmap_img=Pixmap_img.scaled(600,600)
            
        elif file_name[0].endswith(".jpeg") :

            try:
                Pil_img=ImagePil.open(file_path)
                
                
                global image_cv
                        #####using Pil library to open the image#####
                Pixmap_img=QPixmap(file_path)
                       ####using QPixmap to plot the image on a label####
                image_cv=cv2.imread(file_path,0)
                Bit_Depth=Pil_img.bits *(len(Pil_img.getbands()))   ###getting bet depth###
            except:
                self.errorMssg("image is corrupted.")
        elif file_name[0].endswith(".bmp") or file_name[0].endswith(".png"):
            #try:

                Pil_img=ImagePil.open(file_path)
                Pixmap_img=QPixmap(file_path)        ####using QPixmap to plot the image on a label####
                image_cv=cv2.imread(file_path,0)
                Bit_Depth=round((os.path.getsize(file_path)*8)/(Pil_img.width*Pil_img.height))
                #Bit_Depth=Pil_img.bits *(len(Pil_img.getbands()))
                self.mono(Pil_img) 
            #except:
             #   self.errorMssg("image is corrupted.")
        #self.Historgram1()
        
        
        


            
        ### A Dictionary to store Image data #### 
        
        if file_name[0].endswith(".jpeg") or file_name[0].endswith(".png") or file_name[0].endswith(".bmp") :
            Image_data= [{"property":"Width","value":Pil_img.width},{"property":"Height","value":Pil_img.height},
            {"property":"Size","value":(os.stat(file_path).st_size)*8},{"property":"BitDepth","value":Bit_Depth},
            {"property":"Image Color","value":Pil_img.mode},{"property":"format","value":Pil_img.format}]
            row=0
            for i in Image_data:     ####for plotting the data in the table#### 
                self.BasicInfo_tableWidget.setItem(row,0,QtWidgets.QTableWidgetItem(i["property"]))
                self.BasicInfo_tableWidget.setItem(row,1,QtWidgets.QTableWidgetItem(str(i["value"])))
                row=row+1
        
        self.Label.setPixmap(Pixmap_img)
        self.resize(Pixmap_img.width(),Pixmap_img.height())
        
    def setDicomImage(self,file_path):
    
            try:
                global ds
                ds = pydicom.dcmread(file_path)
                new_image=ds.pixel_array.astype(float)  ###extracting the image (pixel array)###
                global scaled_image

                scaled_image= (np.maximum(new_image,0)/new_image.max())*255.0 ####rescale pixels###
                scaled_image=np.uint8(scaled_image) ###convert into 8 bits unsigned integers###
                final_image=ImagePil.fromarray(scaled_image) ###form a pil image###
                Dicom_data=[{"property":"Number of Rows","value":ds.Rows},{"property":"Number of Columns","value":ds.Columns},
                {"property":"Size","value":(os.stat(file_path).st_size)*8},{"property":"BitDepth","value":ds.BitsStored},
                {"property":"Patient Name:","value":ds.get('PatientName','N/A')},{"property":"Patient ID:","value":ds.get('PatientID','N/A')},
                {"property":"Modality","value":ds.get('Modality','N/A')},{"property":"Study Description","value":ds.get('StudyDescription','N/A')},
                {"property":"Patient Age:","value":ds.get('PatientAge','N/A')},{"property":"Image Type","value":ds.get('ImageType','N/A')},
                {"property":"Color","value":ds.get('PhotometricInterpretation','N/A')}]
                row =0
                for i in Dicom_data:
                    self.BasicInfo_tableWidget.setItem(row,0,QtWidgets.QTableWidgetItem(i["property"]))
                    self.BasicInfo_tableWidget.setItem(row,1,QtWidgets.QTableWidgetItem(str(i["value"])))
                    row=row+1
                return final_image ###opening the dicom file##
            except:
                self.errorMssg("image is corrupted.")
    def GetScaleFactorValue(self):
        global Scale_Factor
        Scale_Factor=self.ZoomingFactor_doubleSpinBox.value()
        if Scale_Factor <=0:
                return self.errorMssg("Enter a valid number for Zooming")
        else:
            return Scale_Factor
    def Zoom_NearestNeighbourInterpolation(self):
        Scale_Factor=self.GetScaleFactorValue()
        try:
            if file_name[0].endswith(".jpeg") or file_name[0].endswith(".png") or file_name[0].endswith(".bmp") :
                    
                    gray_image = np.copy(image_cv)             #using numpy array for making a gray scale image
                       
            else:
                gray_image= np.copy(scaled_image)
        except:
            self.errorMssg("Please Choose an image First.")
        global w_Old,h_Old
        w_Old, h_Old = gray_image.shape[:2];                       
        xNew = int(w_Old *  Scale_Factor)                             #New Image Cord.
        yNew = int(h_Old *  Scale_Factor)                     

                
        newImage = npy.zeros([xNew, yNew]);                       #Making a Matrix full of zeroes
        
        for i in range(xNew):
            for j in range(yNew):
                newImage[i , j ]= gray_image[ math.floor(i / Scale_Factor),
                                                + math.floor(j / Scale_Factor)]
        #print ('gg',newImage)                                
        newImage = newImage.astype('uint8')
        Pillow_img=ImagePil.fromarray(newImage)
        NearestNeighbour_PixmapImage=Pillow_img.toqpixmap()
        print('mode',Pillow_img.mode)
        self.nnH_lineEdit.setText(str(Pillow_img.height))
        self.nnW_lineEdit.setText(str(Pillow_img.width))
        self.NearestNeighbor_label.setPixmap(NearestNeighbour_PixmapImage)
        
             ##it is a not a completed Linear interpolation function  
    def Zoom_LinearInterpolation(self):
        Scale_Factor=self.ZoomingFactor_doubleSpinBox.value()

        if file_name[0].endswith(".jpeg") or file_name[0].endswith(".png") or file_name[0].endswith(".bmp") or file_name[0].endswith(".dcm"):
            Pillow_img=Pil_img.convert('L')       #converting the image into gray Scale         
            old=np.asarray(Pillow_img)
            rows,cols=old.shape      # taking the parameters of the old image
            if Scale_Factor==2:
                new=np.zeros((int(Scale_Factor*rows-1),int(cols*Scale_Factor-1)))
            elif Scale_Factor==3:
                new=np.zeros((int(Scale_Factor*rows),int(cols*Scale_Factor)))
            elif Scale_Factor==4:
                new=np.zeros((int(Scale_Factor*rows+1),int(cols*Scale_Factor+1)))


            
            new[:,:]=self.resizeLayer(old[:,:])
            new=new.astype(np.uint8)
            #print("new dim:",new.shape)
            #img.imsave('biig.jpeg',new)

            Pil_image=ImagePil.fromarray(new)
            #frame=np.random.normal(size=(200,200))
            #img=pg.ImageItem(frame)                        
            Pil_img2=Pil_image.convert(mode='L')
            Linear_PixmapImage=Pil_img2.toqpixmap()
            self.Linear_label.setPixmap(Linear_PixmapImage)
           
                         
       
    def resizeLayer(self,old):
        Scaling_Factor=int(self.ZoomingFactor_doubleSpinBox.value())

        rows,cols=old.shape
        if (Scaling_Factor ==2):
            rNew=int(Scaling_Factor*rows-1)
            cNew=int(Scaling_Factor*cols-1)
        elif (Scaling_Factor  ==3):
            rNew=int(Scaling_Factor*rows)
            cNew=int(Scaling_Factor*cols)
        elif (Scaling_Factor  ==4):
            rNew=int(Scaling_Factor*rows+1)
            cNew=int(Scaling_Factor*cols+1)
        #rNew=5
        #cNew=5
        new=np.zeros((rNew,cNew))
        #move old points
        #produce vertical values
        #Scale_Factor=2
        Scale_Factor=Scaling_Factor
        new[0:rNew:int(2*math.ceil(Scale_Factor/2)),0:cNew:int(2*math.ceil(Scale_Factor/2))]=old[0:rows,0:cols]
        
        while Scale_Factor>1:

            new[int(math.ceil(Scale_Factor/2)):rNew:int(2*math.ceil(Scale_Factor/2)),:]=(new[0:rNew-1:int(2*math.ceil(Scale_Factor/2)),:]+ new[int(2*math.ceil(Scale_Factor/2)):rNew:int(2*math.ceil(Scale_Factor/2)),:])/2

        #produce horizontal values
            new[:,int(math.ceil(Scale_Factor/2)):cNew:int(2*math.ceil(Scale_Factor/2))]=(new[:,0:cNew-1:int(2*math.ceil(Scale_Factor/2))]+ new[:,2*math.ceil(Scale_Factor/2):cNew:2*math.ceil(Scale_Factor/2)])/2
        #produce center values
            new[int(math.ceil(Scale_Factor/2)):rNew:2*math.ceil(Scale_Factor/2),int(math.ceil(Scale_Factor/2)):cNew:2*math.ceil(Scale_Factor/2)]=(new[0:rNew-2:2*math.ceil(Scale_Factor/2),0:cNew-2:2*math.ceil(Scale_Factor/2)]+
            new[0:rNew-2:2*math.ceil(Scale_Factor/2),2*math.ceil(Scale_Factor/2):cNew:2*math.ceil(Scale_Factor/2)]+new[2*math.ceil(Scale_Factor/2):rNew:2*math.ceil(Scale_Factor/2),0:cNew-2:2*math.ceil(Scale_Factor/2)]+
            new[2*math.ceil(Scale_Factor/2):rNew:2*math.ceil(Scale_Factor/2),2*math.ceil(Scale_Factor/2):cNew:2*math.ceil(Scale_Factor/2)])/4
            Scale_Factor -=1
            print(Scale_Factor)

        return new
##3##3 the completed Liner interpolation function

    def PlotLinearInterpolation(self,test):
        Scale_Factor=self.ZoomingFactor_doubleSpinBox.value()
        #gray_image=np.copy(image_cv)
        
        if file_name[0].endswith(".jpeg") or file_name[0].endswith(".png") or file_name[0].endswith(".bmp") :
                    
                    gray_image = np.copy(image_cv)             #using numpy array for making a gray scale image
            
        else:
                gray_image= np.copy(scaled_image)
        h,w = gray_image.shape[:2];

        NewWidth=int(w*Scale_Factor)   #calculating scale factor
        NewHeight=int(h*Scale_Factor)
        # gray_image=np.copy(test)
        # NewWidth=6
        # NewHeight=6
        WW= self.im_interp(gray_image.T,NewHeight,NewWidth)

        WW= self.im_interp(gray_image,NewHeight,NewWidth)
        #print("aa",WW)
        PillImg=ImagePil.fromarray(WW)
        
        #print('jj',PillImg.mode)              #forming a pil image
        uu=PillImg.convert(mode='L')
        Px=uu.toqpixmap()
        self.lnH_lineEdit.setText(str(PillImg.height))
        self.lnW_lineEdit.setText(str(PillImg.width))

        self.Linear_label.setPixmap(Px)
        
        print("no")
    def interpolation(self,y0,x0, y1,x1, x):
        frac = (x - x0) / (x1 - x0)
        return y0*(1-frac) + y1 * frac    # forming linear interpolation equation 
    def get_coords(self,im, W, H):
        h,w = im.shape
        x = np.arange(1,w+1,1) * W/w 
        y = np.arange(1,h+1,1) * H/h
        #print ("edd",x,y) 
        return x,y
    def im_interp(self,im, H,W):
        X = np.zeros(shape=(H,W))
        x, y = self.get_coords(im, W, H)
        #print('im',im)
        for i,v in enumerate(X):
            y0_idx = np.argmax(y >i) - 1
            #print("YO_idx",y0_idx)
            for j,_ in enumerate(v):
                # subtracting 1 because this is the first val
                # that is greater than j, want the idx before that
                #print ('xAnd J',x,j)
                x0_idx = np.argmax(x > j)  -1 #floor
                x1_idx = np.argmax(j < x)       #ceil

                x0 = x[x0_idx]
                x1 = x[x1_idx]
                #print('x0X1',x0,x1)
                y0 = im[y0_idx, x0_idx -1]
                y1 = im[y0_idx, x1_idx -1]
            
                X[i,j] = self.interpolation(y0, x0, y1, x1, j)
        return X    
  
    def SetScaleFactor(self):
        Hnew=int(self.nnH_lineEdit.text())
        #Wnew=int(self.nn_lineEdit.text())
        ScaleFactor=Hnew/h_Old
        self.ZoomingFactor_doubleSpinBox.setValue(ScaleFactor)     
    
    def PlotConstructedImage(self):
        Rows=128
        Columns=128
        # global X
        # X =np.zeros([Rows,Columns])
        # for i in range(Columns):
        #     for j in range (Rows):
        #         if 29 <= j < 49:
        #             if 29 <= i < 99:
        #                 X[j,i]= 1
        #         elif 49 <= j < 99:
        #             if 54 <= i <74:
        #                 X[j,i]=1
        #         else:
        #             X[j,i]=0
        self.Constructed_Image=npy.zeros([Rows,Columns])
        self.Constructed_Image[28:48,28:98]=255
        self.Constructed_Image[48:98,53:73]=255
        self.Constructed_Image=self.Constructed_Image.astype('uint8')            
        self.graphicsView.canvas.axes.clear()
        self.graphicsView.canvas.axes.imshow(self.Constructed_Image)
        self.graphicsView.canvas.draw()


    def naive_image_rotate(self,imgg, angle_degree):
        
          

            ########## Changing our angle from degrees into radians ######## 
            rads = math.radians(angle_degree)
            self.Constructed_Image=np.asarray(imgg)
            #####forming the new image with the same size of the constructed image######
            rot_img = np.uint8(np.zeros(self.Constructed_Image.shape))
        
            
            height = rot_img.shape[0]
            width  = rot_img.shape[1]
            #### Finding the center point of rotated (or original) image. using floor division###
            midx,midy = (width//2, height//2)

            Text=str(self.comboBox.currentText())###knowing the user choice####
            for i in range(rot_img.shape[0]):
                for j in range(rot_img.shape[1]):
                    ##forming Rotation matrix equation for rotating
                    x= (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads)
                    y= -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads)

                    x=x+midx 
                    y=y+midy
                    if Text=="Rotate by Nearest Neighbour":        ######USING NEARESTNEIGHBOUR INTERPOLATION 
                        if (round(x)>=0 and round(y)>=0 and round(x)<=self.Constructed_Image.shape[0]-1 and  round(y)<=self.Constructed_Image.shape[1]-1):
                            
                            rot_img[i][j] = self.Constructed_Image[round(x)][round(y)]
                        else:
                            rot_img[i][j] =255
                    elif Text=="Rotate by Linear interpolation": ###3USING LINEAR INTERPOLATION

                        x_floor = math.floor(x)
                        x_ceil  = math.ceil(x)
                        y_floor = math.floor(y)
                        y_ceil  = math.ceil(y)
                
                    
                        if (0<=x_floor<=height-1) and (0<=y_floor<=width-1) and (0<=x_ceil<=height-1) and (0<=y_ceil<=width-1):
                          
                            if (x_ceil == x_floor) and (y_ceil == y_floor):
                                
                                q = self.Constructed_Image[x_floor][y_floor]
                            elif (x_ceil == x_floor):
                                #inbetween 2 values only in y----1 linear interpolarion
                                q1 = self.Constructed_Image[x_floor][y_floor]
                 
                                q2 = self.Constructed_Image[x_floor][ y_ceil]
                      
                                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
                            elif (y_ceil == y_floor):
                                #inbetween 2 values only in x-----1 linear interpolation
                                q1 = self.Constructed_Image[x_floor][y_ceil]
                                
                                q2 = self.Constructed_Image[x_ceil][y_ceil]
                                
                                q = q1 * (x_ceil - x) + q2	 * (x - x_floor)
                            else:
                                
                                v1 = self.Constructed_Image[x_floor][ y_floor]
                                v2 = self.Constructed_Image[x_ceil][ y_floor]
                                v3 = self.Constructed_Image[x_floor][ y_ceil]
                                v4 = self.Constructed_Image[x_ceil][ y_ceil]
                    
                                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                                q2 = v3 * (x_ceil -x)+ v4 * (x - x_floor)
                    
                                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

                            rot_img[i][j] = q
        
                        else:
                            rot_img[i][j]=255
                        

            return rot_img 
     
    
    def naive_shearing(self):
        try:

            '''
            This function shears the image around its center by amount of degrees
            provided. The size of the rotated image is same as that of original image.
            '''
            # First we will convert the degrees into radians
            #rads = math.radians(45/2)

            # FORMING THE SHEARED IMAGE WITH THE SAME SIZE OF THE ORIGINAL IMAGE
            rot_img = np.uint8(np.zeros(self.Constructed_Image.shape))

            # GETTING HEIGHT AND WIDTH
            height = rot_img.shape[0]
            width  = rot_img.shape[1]
            # GETTING THE CENTER POINT
            midx,midy = (width//2, height//2)
            for i in range(rot_img.shape[0]):
                for j in range(rot_img.shape[1]):
        
                    x=i-midx
                    y=(i-midx)-(j-midy)
                    x=x+midx
                    y=y+midy
                    if (round(x)>=0 and round(y)>=0 and round(x)<=self.Constructed_Image.shape[0]-1 and  round(y)<=self.Constructed_Image.shape[1]-1):
                        
                        rot_img[i][j] = self.Constructed_Image[round(x)][round(y)]
                    else:
            
                        rot_img[i][j] =0
                    ### Drawing the image on canvas     
            self.Rotation_graphicsView.canvas.axes.clear()
            self.Rotation_graphicsView.canvas.axes.imshow(rot_img,cmap='gray')
            self.Rotation_graphicsView.canvas.draw() 
        except:
              self.errorMssg("There is a problem In your Steps, maybe you didn't press the Apply button First")           

    def rotate(self):
        #try:

       ##
            AngleValue=int(self.lineEdit.text())
            rotated_image = self.naive_image_rotate(45)
    
            self.Rotation_graphicsView.canvas.axes.clear()
            self.Rotation_graphicsView.canvas.axes.imshow(rotated_image,cmap='gray')
            self.Rotation_graphicsView.canvas.draw()
        #except:
           #   self.errorMssg("There is a problem In your Steps, maybe you didn't press the Apply button or you didn't enter the angle rotation")           
    def ImageRotate(self,Image ,rotation_amount_degree):
        Image=np.asarray(Image)
        rotation_amount_rad = rotation_amount_degree * np.pi / 180.0
        (width,height) = Image.shape
        width=int(width)
        height=int(height)
        rotated_image = np.zeros((width,height))

        rotated_height, rotated_width = Image.shape
        mid_row = math.floor( (rotated_height+1)/2 )
        mid_col = math.floor( (rotated_width+1)/2 )


        for r in range(rotated_height):
            for c in range(rotated_width):
                #  apply rotation matrix, the other way
                y = (r-mid_col)*math.cos(rotation_amount_rad) + (c-mid_row)*math.sin(rotation_amount_rad)
                x = -(r-mid_col)*math.sin(rotation_amount_rad) + (c-mid_row)*math.cos(rotation_amount_rad)

                #  add offset
                y += mid_col
                x += mid_row
                x = round(x)
                y = round(y)

                if (x >= 0 and y >= 0 and x < rotated_width and y < rotated_height):
                    rotated_image[r][c] = Image[y][x]
        return rotated_image


    def SetInformationDirection(self):
        self.Direction_lineEdit.show()
        AngleValue=int(self.lineEdit.text())
        if AngleValue > 0:
            self.Direction_lineEdit.setText("The Rotation Direction in anticlockwise")
        elif AngleValue < 0:
            self.Direction_lineEdit.setText("The Rotation Direction in clockwise")
        else:
            self.Direction_lineEdit.setText("No Rotation")

    def Historgram1(self):
        try:
        # first we converted the image to gray scale
            Pil_imggg=Pil_img.convert(mode='L')
        # drawing the image    
            self.image1_graphicsView.canvas.axes.clear()
            self.image1_graphicsView.canvas.axes.imshow(Pil_imggg,cmap='gray')
            self.image1_graphicsView.canvas.draw()
        # convert the image into numpy array    
            img=np.asarray(Pil_imggg)
        #putting pixels into 1 D array
            self.flat=img.flatten()
        # calling plothistogram function     
            self.plothistogram(Pil_imggg)
            hist =self.get_histogram(self.flat, 256)
            cs = self.cumsum(hist)
            #print('cs',cs)
            nj = (cs ) * 255
            #print('nj',nj)
            N = cs.max() - cs.min()
            #print('cs after max -min')
            # re-normalize the cdf
            cs = nj / N
           # casting it back to uint8 since
            cs = cs.astype('uint8')

            # get the value from cumulative sum for every index in flat, and set that as img_new
            img_new = cs[self.flat]
            #img_new=ImagePil.fromarray(img_new)
            self.plothistogram2(img_new)
            #plt.hist(img_new, bins=50)
            img_new = np.reshape(img_new, img.shape)
            self.image2_graphicsView.canvas.axes.clear()
            self.image2_graphicsView.canvas.axes.imshow(img_new,cmap='gray')
            self.image2_graphicsView.canvas.draw()
        except:
                
              self.errorMssg("Browse an image first")      


        
   
    def get_histogram(self,image, bins):
    # array with size of bins, set to zeros
            histogram = np.zeros(bins)
            
            # loop through pixels and sum up counts of pixels
            for pixel in image:
                histogram[pixel] += 1
            
            # return our final result
            return histogram

    def cumsum(self,a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

#function to plot histogram
    def plothistogram(self,img):
        pixels=[]  #create list of values 0-255
        for x in range(256):
             pixels.append(x)
        #initialize width and height of image
        width,height=img.size
        counts=[]
        #for each intensity value
        for i in pixels:
            #set counter to 0
            num=0

            #traverse through the pixels
            for x in range(width):
                for y in range(height):

                #if pixel intensity equal to intensity level
                #increment counter
                    if (img.getpixel((x,y))==i):
                        num=num+1

            #append frequency of intensity level 
            counts.append(num)
        Count=np.array(counts)
        self.histogram1_graphicsView.canvas.axes.clear()
        #self.plothistogram(Pil_imgggg)
        self.histogram1_graphicsView.canvas.axes.bar(pixels,Count)
        self.histogram1_graphicsView.canvas.draw()
        self.originalhistogramnormalization_graphicsView.canvas.axes.clear()
        #self.plothistogram(Pil_imgggg)
        self.originalhistogramnormalization_graphicsView.canvas.axes.bar(pixels,Count/(width*height),color="orange")
        self.originalhistogramnormalization_graphicsView.canvas.draw()
        self.originalhistogramnormalization_graphicsView.canvas.axes.set_xlabel('intenisty value',color='white')
        self.originalhistogramnormalization_graphicsView.canvas.axes.set_ylabel('Probability',color='white')

    def plothistogram2(self,img):
        pixels=[]
        imgg=ImagePil.fromarray(img)
        #create list of values 0-255
        for x in range(256):
             pixels.append(x)
        
        #initialize width and height of image
        width,height=imgg.size
        counts=[]
        for i in pixels:
            #set counter to 0
            num=0
            for x in range(width):
                for y in range(height):
                    if (imgg.getpixel((x,y))==i):
                        num=num+1

            #append frequency of intensity level 
            counts.append(num)

        Count=np.array(counts)    
        self.histogram2_graphicsView.canvas.axes.clear()
        self.histogram2_graphicsView.canvas.axes.bar(pixels,Count)
        self.histogram2_graphicsView.canvas.draw()
        self.equilizedhistogramnormalization_graphicsView.canvas.axes.clear()
        self.equilizedhistogramnormalization_graphicsView.canvas.axes.bar(pixels,Count/(width*height),color="orange")
        self.equilizedhistogramnormalization_graphicsView.canvas.draw()
        self.equilizedhistogramnormalization_graphicsView.canvas.axes.set_xlabel('intenisty value',color='white')
        self.equilizedhistogramnormalization_graphicsView.canvas.axes.set_ylabel('Probability',color='white')

    def average_kernel(self,kernal_size):
 
        kernel = np.ones([kernal_size,kernal_size])
        kernel = np.multiply(kernel,1/kernel.sum())
        return kernel

    def convolution2D(self,image, kernel):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))
        
        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]
        padding= int(xKernShape/2)  
        strides=1
        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        #3xOutput=round ((xImgShape+xKernShape-1+(2*padding)))
        ##yOutput= round(yImgShape+yKernShape-1+(2*padding))
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        else:
            imagePadded = image

        # Iterate through image
        for y in range(imagePadded.shape[1]):
            # Exit Convolution
            if y > imagePadded.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(imagePadded.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > imagePadded.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            #output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return output
    def convolution2d(self,image, kernel):
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]
        padded_img = np.zeros((yImgShape + ( yKernShape - 1), xImgShape + (xKernShape - 1)))
        padded_img[( yKernShape // 2):-( yKernShape // 2), (xKernShape// 2):-(xKernShape // 2)] = image
        m, n = kernel.shape
        if (m == n):
            y, x = padded_img.shape
            y = y - m + 1
            x = x - m + 1
            new_image = np.zeros((y,x))
            for i in range(y):
                for j in range(x):
                    new_image[i][j] = np.sum(padded_img[i:i+m, j:j+m]*kernel)
        return new_image    
    

    def BoxKernel(self):
            try:
                Pil_imggg=Pil_img.convert(mode='L')
                        # drawing the image    
                self.unmaskedImage_graphicsView.canvas.axes.clear()
                self.unmaskedImage_graphicsView.canvas.axes.imshow(Pil_imggg,cmap='gray')
                self.unmaskedImage_graphicsView.canvas.draw()
            # convert the image into numpy array    
                img=np.asarray(Pil_imggg)
                 ##imgg=np.asarray(Pil_img)
                # calling functions
                kernal_size=int(self.Kernelsize_doubleSpinBox.value())
                kernal=self.average_kernel(kernal_size)
                MaskedImage=self.convolution2D(img,kernal)
                self.masked=MaskedImage
                SubtractedImage=img-MaskedImage

                ImageAfterKfactor=int(self.kfactor_doubleSpinBox.value())*SubtractedImage
                ImageAfterKfactor=img + ImageAfterKfactor
                ImageAfterKfactor=self.normalization(ImageAfterKfactor)
                #nj = (ImageAfterKfactor ) * ((2**3)-1)
                ##print("f",Pil_img.bits)
                #N = ImageAfterKfactor.max() - ImageAfterKfactor.min()
                #ImageAfterKfactor=((2**Pil_img.bits)-1)*((ImageAfterKfactor-np.min(ImageAfterKfactor))/(np.max(ImageAfterKfactor)-np.min(ImageAfterKfactor)))
                #ImageAfterKfactor = nj / N
                #scaled_img = (np.double(enhanced_img) - np.double(np.min(enhanced_img)))/ (np.double(np.max(enhanced_img))
                  #- np.double(np.min(enhanced_img))) * np.double(np.max(enhanced_img))
                
                self.maskedImage_graphicsView.canvas.axes.clear()
                self.maskedImage_graphicsView.canvas.axes.imshow(ImageAfterKfactor,cmap='gray')
                self.maskedImage_graphicsView.canvas.draw()
            except:
                    self.errorMssg("choose an image, enter k value and an odd number for kernel size first")      
    def normalization(self,data):
                for i in range (data.shape[0]):
                    for j in range (data.shape[1]):
                        if data [i,j]> ((2**8)-1):
                            data[i,j]=((2**8)-1)
                        elif data[i,j]<0:
                            data[i,j]=0
                return data

    def SandP(self):
        try:

            Pil_imggg=Pil_img.convert(mode='L')
            
            self.Noiseyimg=np.asarray(Pil_imggg)
        
        # Getting the dimensions of the image
            row , col = self.Noiseyimg.shape
            avg_size=int((row+col)/2)
            # 5% to 96% 
            number_of_pixels = random.randint(0,avg_size )
            for i in range(number_of_pixels):
            
                # Pick a random y coordinate
                y_coord=random.randint(0, row - 1)
                
                # Pick a random x coordinate
                x_coord=random.randint(0, col - 1)
                
                # Color that pixel to white
                self.Noiseyimg[y_coord][x_coord] = 255
      
            number_of_pixels = random.randint(0 ,avg_size )
            for i in range(number_of_pixels):
            
                # Pick a random y coordinate
                y_coord=random.randint(0, row - 1)
                
                # Pick a random x coordinate
                x_coord=random.randint(0, col - 1)
                self.Noiseyimg[y_coord][x_coord] = 0
            self.saltpaper_graphicsView.canvas.axes.clear()
            self.saltpaper_graphicsView.canvas.axes.imshow(self.Noiseyimg,cmap='gray')
            self.saltpaper_graphicsView.canvas.draw()    
        except:
                    self.errorMssg("Browse an image first")      

           
           
    def median_filter(self):
        try:

            data=self.Noiseyimg
            filter_size=3
            temp = []
            indexer = filter_size // 2
            print("ind",indexer)
            data_final = []
            data_final = np.zeros((len(data),len(data[0])))
            for i in range(len(data)):

                for j in range(len(data[0])):

                    for z in range(filter_size):
                        if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                            for c in range(filter_size):
                                temp.append(0)
                        else:
                            if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                                temp.append(0)
                            else:
                                for k in range(filter_size):
                                    temp.append(data[i + z - indexer][j + k - indexer])

                    temp=self.bubble_sort(temp)
                    data_final[i][j] = temp[len(temp) // 2]
                    temp = []
            self.saltpaper_graphicsView.canvas.axes.clear()
            self.saltpaper_graphicsView.canvas.axes.imshow(data_final,cmap='gray')
            self.saltpaper_graphicsView.canvas.draw()        
        except:
                    self.errorMssg("Browse an image first")          ##return data_final       

    def bubble_sort(self,nums):
    # We set swapped to True so the loop looks runs at least once
        swapped = True
        while swapped:
            swapped = False
            for i in range(len(nums) - 1):
                if nums[i] > nums[i + 1]:
                    # Swap the elements
                    nums[i], nums[i + 1] = nums[i + 1], nums[i]
                    # Set the flag to True so we'll loop again
                    swapped = True        # then move on to the next number in the list
        return nums    
        ##print('After sorting: ',numSort) 

    def fft(self):
        fft_image=Pil_img.convert(mode='L')
        self.FFT_widget.canvas.axes.clear()
        self.FFT_widget.canvas.axes.imshow(fft_image,cmap='gray')
        self.FFT_widget.canvas.draw()   
        fft_image=np.fft.fftshift(np.fft.fft2(fft_image))
        phaseImage=np.arctan2(fft_image.imag,fft_image.real)
        MagnitudeImage=np.sqrt(fft_image.real**2+fft_image.imag**2)
        #c = 255 / np.log(1 + np.max(MagnitudeImage))
        #magnitude = c * np.log (1 + MagnitudeImage)
        #log_magnitude=np.asarray(magnitude)
        PhaseImageAfterLog=np.log(phaseImage+2*math.pi)
        MagnitudeImageAfterLog=np.log(MagnitudeImage+1)


        self.Phase_widget.canvas.axes.clear()
        self.Phase_widget.canvas.axes.imshow(phaseImage,cmap='gray')
        self.Magnitude_widget.canvas.draw()   
        self.Magnitude_widget.canvas.axes.clear()
        self.Magnitude_widget.canvas.axes.imshow(MagnitudeImage,cmap='gray')
        self.Magnitude_widget.canvas.draw()   
        self.PhaseLog_widget.canvas.axes.clear()
        self.PhaseLog_widget.canvas.axes.imshow(PhaseImageAfterLog,cmap='gray')
        self.PhaseLog_widget.canvas.draw()
        self.MagniduteLog_widget.canvas.axes.clear()
        self.MagniduteLog_widget.canvas.axes.imshow(MagnitudeImageAfterLog,cmap='gray')
        self.MagniduteLog_widget.canvas.draw()   

    def FourierFilter(self):
        try:
                #``````````````````````gettting our image from browsing``````````````````````# 
                original_img=Pil_img.convert(mode="L")
                original_img=np.asarray(original_img)
                #'''''''''''''''''''''making our box kernel`````````````````#
                Boxkernel=self.average_kernel(int(self.doubleSpinBox.value()))
                #````````````````````padding out kernel and image to be equal in size`````````````#    
                padded_kernel,padded_image = self.kernel_padding(original_img,Boxkernel)
                #````````````````````doing our convolution in spatial domain for low pass filter``````#
                BluredImage_fromSpatial=self.convolution2D(original_img,Boxkernel)
                #``````````````making fft for the image and the kernel````````#       
                padded_image=np.fft.fft2(padded_image)
                padded_kernel=np.fft.fft2(padded_kernel)
        
                fft_lowpass = np.multiply(np.fft.fftshift(padded_kernel),np.fft.fftshift(padded_image))
                output_lowpass = np.fft.ifft2(np.fft.ifftshift(fft_lowpass))
                output_lowpass = np.fft.fftshift(output_lowpass)
                output_lowpass=np.abs(output_lowpass)
                eewe,BluredImage_fromSpatial=self.kernel_padding(BluredImage_fromSpatial,Boxkernel)
                image_filter =  output_lowpass-BluredImage_fromSpatial
                # nj = (image_filter - image_filter.min()) * 255
                # N = image_filter.max() - image_filter.min()
                # image_filter = nj / N
                image_filter=self.normalization(image_filter)
                BluredImage_fromSpatial=self.normalization(BluredImage_fromSpatial)
                output_lowpass=self.normalization(output_lowpass)
                #print("ss",image_filter)       
                self.originalImage_widget.canvas.axes.clear()
                self.originalImage_widget.canvas.axes.imshow(original_img,cmap='gray')
                self.originalImage_widget.canvas.draw() 
                self.inspatial_widget.canvas.axes.clear()
                self.inspatial_widget.canvas.axes.imshow(BluredImage_fromSpatial,cmap='gray')
                self.inspatial_widget.canvas.draw() 
                self.fourierdomain_widget.canvas.axes.clear()
                self.fourierdomain_widget.canvas.axes.imshow(output_lowpass,cmap='gray')
                self.fourierdomain_widget.canvas.draw()   
                self.subtraction_widget.canvas.axes.clear()
                self.subtraction_widget.canvas.axes.imshow(image_filter,cmap='gray')
                self.subtraction_widget.canvas.draw()
        except:
            
              self.errorMssg("An error has occurred")

    
    def kernel_padding(self,imagee,kernel):
        
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))
        
        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = imagee.shape[0]
        yImgShape = imagee.shape[1]
        
        if  (xImgShape % 2 == 0):
            if  (yImgShape % 2 == 0):
                odd_padding = np.zeros((imagee.shape[0] + 1,imagee.shape[1]+1))
                print(odd_padding.shape)
                odd_padding[:int(-1 * 1), :int(-1 * 1)] = imagee
            else:
                odd_padding = np.zeros((imagee.shape[0]+1,imagee.shape[1]))
                odd_padding[:int(-1 * 1), :] = imagee
        else:
            if (yImgShape % 2 == 0):
                odd_padding = np.zeros((imagee.shape[0],imagee.shape[1]+1))
                odd_padding[:,:int(-1 * 1)] = imagee
            else:
                odd_padding = imagee
            
            
        xImgShape = odd_padding.shape[0]
        yImgShape = odd_padding.shape[1]
        
        padding_x = int((xImgShape-xKernShape) / 2)
        padding_y = int((yImgShape-yKernShape) / 2 )
        strides=1
        # Shape of Output Convolution
        

        # Apply Equal Padding to All Sides
        if padding_x != 0 or padding_y != 0:
            kernalPadded = np.zeros((xKernShape + padding_x*2, yKernShape + padding_y*2))
            kernalPadded[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)] = kernel
            
        else:
            kernalPadded = kernel
        
        return (kernalPadded, odd_padding)
    def notch_reject_filter(self,shape, d0=9, u_k=0, v_k=0):
        P, Q = shape
        # Initialize filter with zeros
        H = np.zeros((P, Q))

        # Traverse through filter
        for u in range(0, P):
            for v in range(0, Q):
                # Get euclidean distance from point D(u,v) to the center
                D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
                D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

                if D_uv <= d0 or D_muv <= d0:
                    H[u, v] = 0.0
                else:
                    H[u, v] = 1.0

        return H
#-----------------------------------------------------
    def removepattern(self): 
        img = Pil_img.convert(mode="L")
        img=np.asarray(img)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        phase_spectrumR = np.angle(fshift)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        img_shape = img.shape

        H1 = self.notch_reject_filter(img_shape, 4, 38, 30)
        H2 = self.notch_reject_filter(img_shape, 4, -42, 27)
        H3 = self.notch_reject_filter(img_shape, 2, 80, 30)
        H4 = self.notch_reject_filter(img_shape, 2, -82, 28)

        NotchFilter = H1*H2*H3*H4
        NotchRejectCenter = fshift * NotchFilter 
        NotchReject = np.fft.ifftshift(NotchRejectCenter)
        inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result


        Result = np.abs(inverse_NotchReject)
        self.widget.canvas.axes.clear()
        self.widget.canvas.axes.imshow(img,cmap='gray')
        self.widget.canvas.draw()
        self.widget_2.canvas.axes.clear()
        self.widget_2.canvas.axes.imshow(Result,cmap='gray')
        self.widget_2.canvas.draw()
        
    def fourier2(self):
        img = Pil_img

        # do dft saving as complex output
        dft = np.fft.fft2(img, axes=(0,1))

        # apply shift of origin to center of image
        dft_shift = np.fft.fftshift(dft)

        # generate spectrum from magnitude image (for viewing only)
        mag = np.abs(dft_shift)
        spec = np.log(mag) / 20

        # create circle mask
        radius = int(self.horizontalSlider.value())

        mask = np.zeros_like(img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

        # blur the mask
        mask2 = cv2.GaussianBlur(mask, (19,19), 0)

        # apply mask to dft_shift
        dft_shift_masked = np.multiply(dft_shift,mask) / 255
        dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift)
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


        # do idft saving as complex output
        img_back = np.fft.ifft2(back_ishift, axes=(0,1))
        img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
        img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)
        img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.uint8)

        self.mask1_widget.canvas.axes.clear()
        self.mask1_widget.canvas.axes.imshow(mask,cmap='gray')
        self.mask1_widget.canvas.draw()
        self.mask2_widget.canvas.axes.clear()
        self.mask2_widget.canvas.axes.imshow(mask2,cmap='gray')
        self.mask2_widget.canvas.draw()
        self.imagemasked1_widget.canvas.axes.clear()
        self.imagemasked1_widget.canvas.axes.imshow(img_filtered,cmap='gray')
        self.imagemasked1_widget.canvas.draw()
        self.imagemasked2_widget.canvas.axes.clear()
        self.imagemasked2_widget.canvas.axes.imshow(img_filtered2,cmap='gray')
        self.imagemasked2_widget.canvas.draw()
      

    def phantom_gen (self) :   
            # 50, 120, 200
        x = np.linspace(-10, 10, 256)
        y = np.linspace(-10, 10, 256)
        x, y = np.meshgrid(x, y)
        x_0 = 0
        y_0 = 0
        mask = np.sqrt((x-x_0)**2+(y-y_0)**2)

        r = 5
        for x in range(256):
            for y in range(256):
                if mask[x,y] < r:
                    mask[x,y] = 80
                elif mask[x,y] >= r:
                    mask[x,y] = 0




        squares = np.full((256, 256), 50)
        for i in range(35, 221):
            for j in range(35, 221):
                squares[i,j] = 120

        self.final = squares + mask 
        self.widget_3.canvas.axes.clear()
        self.widget_3.canvas.axes.imshow(self.final,cmap='gray')
        self.widget_3.canvas.draw()
        #return(final)    
    def noisy(self):
        image=self.final
        if self.radioButton.isChecked():
            x, y = image.shape
            mean = 0
            sigma = 5
            n = np.random.normal(loc=mean, scale=sigma, size=(x,y))
            self.noisy = image + n
            #self.noisy=np.clip(self.noisy,0,255)
            self.widget_4.canvas.axes.clear()
            self.widget_4.canvas.axes.imshow(self.noisy,cmap='gray')
            self.widget_4.canvas.draw()
        elif self.radioButton_2.isChecked():
            x, y = image.shape
            a = -10
            b = 10
            n = np.zeros((x,y), dtype=np.float64)
            for i in range(x):
                for j in range(y):
                    n[i][j] = np.random.uniform(a,b)
            self.noisy = image + n
                #self.noisy=np.clip(self.noisy,0,255)

            self.widget_4.canvas.axes.clear()
            self.widget_4.canvas.axes.imshow(self.noisy,cmap='gray')
            self.widget_4.canvas.draw()    
    def selectroi(self):
            #image_path
        #img_path="image.jpeg"

        #read image
        #img_raw = cv2.imread(img_path)

        #select ROI function
        rs = RectangleSelector(self.widget_4.canvas.axes, self.line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)
        # self.widget_7.canvas.axes.clear()
        # self.widget_7.canvas.axes.imshow(rs,cmap='gray')
        # self.widget_7.canvas.draw()                  
        roi = cv2.selectROI(self.noisy)
        
        # self.widget_4.canvas.axes.clear()
        # self.widget_4.canvas.axes.imshow(roi,cmap='gray')
        # self.widget_4.canvas.draw()    
        #print rectangle points of selected roi
        print("ee",roi)

        #Crop selected roi from raw image
        roi_cropped = self.noisy[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

        #show cropped image
        cv2.imshow("ROI", roi_cropped)

        cv2.imwrite("crop.jpeg",roi_cropped)
        #ROI=np.asarray(roi_cropped)
        ROI=ImagePil.fromarray(roi_cropped)
        #self.plothistogramROI(ROI)
        #hold window
        cv2.waitKey(0)    
    
    def plothistogramROI(self,img):
        
        pixels=[]  #create list of values 0-255
        for x in range(256):
             pixels.append(x)
        #initialize width and height of image
        width,height=img.size
        print("width:",width,"h:",height)
        counts=[]
        #for each intensity value
        for i in pixels:
            #set counter to 0
            num=0
            #traverse through the pixels
            for x in range(width):
                for y in range(height):

                #if pixel intensity equal to intensity level
                #increment counter
                    if (img.getpixel((x,y))==i):
                        num=num+1
                
            #append frequency of intensity level 
            counts.append(num)
        Count=np.array(counts)
        self.widget_5.canvas.axes.clear()
        self.widget_5.canvas.axes.bar(pixels,Count)
        self.widget_5.canvas.draw()
    def histoROI(self,img):
        arr_1D=img.reshape(-1)
        max=int(np.amax(img))
        max_bit_depth=int(2**(np.ceil(np.log2(max+1))))
        histogram=np.zeros(max_bit_depth)
        pixel_val=np.arange(max_bit_depth)
        for i in range (len(arr_1D)):
            histogram[int(arr_1D[i])] +=1
        normalized=histogram/len(arr_1D)
        self.mean=0
        sum_square_diff=0
        for i in range(max_bit_depth):
            self.mean += normalized[i] * pixel_val[i]
            sum_square_diff += ((pixel_val[i]-self.mean)**2)*normalized[i]
        self.var=sum_square_diff/len(arr_1D)
        self.st=np.sqrt(self.var)
        self.widget_5.canvas.axes.clear()
        self.widget_5.canvas.axes.bar(pixel_val,normalized)
        self.widget_5.canvas.draw()                    
    def line_select_callback(self,eclick, erelease):
        self.widget_4.canvas.axes.clear() 
        self.widget_4.canvas.axes.imshow(self.noisy,cmap='gray')
        self.widget_4.canvas.draw()            
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
        print("rect",rect)
        print(x1,x2,y1,y2)
        self.noisyy=self.noisy[round(y1):round(y2),round(x1): round(x2)]
        self.noisyy=self.normalization(self.noisyy)
        self.widget_7.canvas.axes.clear()
        self.widget_7.canvas.axes.imshow(self.noisyy,cmap='gray')
        self.widget_7.canvas.draw()        
        Pillow=ImagePil.fromarray(self.noisyy)
        
        #Pillow=Pillow.crop((x1-x2,x2,y1,y2))
        self.histoROI(self.noisyy)
        #Pixmap_img=Pillow.toqpixmap()
        #self.label_31.setPixmap(Pixmap_img)
        mean=np.mean(self.noisyy)
        var=np.var(self.noisyy)
        st=np.sqrt(var)
        self.label_25.setText(str(self.mean))
        self.label_26.setText(str(self.st))
    def B_P_function(self): 
        self.B_P_phantom = shepp_logan_phantom()
        self.B_P_phantom = rescale(self.B_P_phantom, scale=0.64, mode='reflect', channel_axis=None)
   
        theta_b = range(0,180)
        theta_a = [0, 20, 40, 60,80,100,120,140, 160]
        theta = np.linspace(0., 180., max(self.B_P_phantom.shape), endpoint=False)
        #sinogram = radon(self.B_P_phantom, theta=theta)
        sinogram=self.discrete_radon_transform(self.B_P_phantom,theta)
        
        #dx, dy = 0.5 * 180.0 / max(self.B_P_phantom.shape), 0.5 / sinogram.shape[0]
        self.Schepp_widget.canvas.axes.clear()
        self.Schepp_widget.canvas.axes.imshow(self.B_P_phantom,cmap='gray')
        self.Schepp_widget.canvas.draw()   
        # self.B_P_phantom_fig.fig1.figimage(self.B_P_phantom,cmap='gray')
        self.Sinogram_widget.canvas.axes.clear()
        
        #self.Sinogram_widget.canvas.axes.imshow(sinogram, cmap='gray',extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy)
        #                                ,aspect='auto')
                # self.B_P_sinogram_fig.draw()
        sinogramm=np.array(sinogram)
        sinogram=np.rot90(np.rot90(np.rot90(sinogramm)))
        self.Sinogram_widget.canvas.axes.imshow(sinogram,cmap='gray')
        
        self.Sinogram_widget.canvas.draw()   

  
        if (self.comboBox_2.currentText() == 'With Angle 0, 20,40,--'):
    #        sinogram = radon(self.B_P_phantom, theta=theta_a)
            sinogram=self.discrete_radon_transform(self.B_P_phantom,theta_a)

            reconstruction_fbp = iradon(sinogram, theta=theta_a, filter_name = None)

        elif(self.comboBox_2.currentText() == 'No Filter'):
            sinogram = self.discrete_radon_transform(self.B_P_phantom, theta=theta_b)
            reconstruction_fbp = iradon(sinogram, theta = theta_b, filter_name = None)   
        elif (self.comboBox_2.currentText() == 'Roo7 ll box el tany'):
            sinogram = self.discrete_radon_transform(self.B_P_phantom, theta=theta_b)
            reconstruction_fbp = iradon(sinogram, theta=theta_b, filter_name = self.comboBox_3.currentText() )
        reconstruction_fbp=np.flip(reconstruction_fbp)
        reconstruction_fbp=np.flip(reconstruction_fbp,1)    


        self.Laminogram_widget.canvas.axes.clear()
        self.Laminogram_widget.canvas.axes.imshow(reconstruction_fbp,cmap='gray')
        self.Laminogram_widget.canvas.draw()
    def B_P_functionn(self): 
        self.B_P_phantom = shepp_logan_phantom()
        self.B_P_phantom = rescale(self.B_P_phantom, scale=0.64, mode='reflect', channel_axis=None)

        theta_b = range(0,180)
        theta_a = [0, 20, 40, 60,80,100,120,140, 160]
        theta = np.linspace(0., 180., max(self.B_P_phantom.shape), endpoint=False)
        #sinogram = radon(self.B_P_phantom, theta=theta)
        sinogram=radon(self.B_P_phantom,theta)
        
        #dx, dy = 0.5 * 180.0 / max(self.B_P_phantom.shape), 0.5 / sinogram.shape[0]
        self.Schepp_widgett.canvas.axes.clear()
        self.Schepp_widgett.canvas.axes.imshow(self.B_P_phantom,cmap='gray')
        self.Schepp_widgett.canvas.draw()   
        # self.B_P_phantom_fig.fig1.figimage(self.B_P_phantom,cmap='gray')
        self.Sinogram_widgett.canvas.axes.clear()
        
       
        sinogramm=np.array(sinogram)
        sinogram=np.rot90(np.rot90(np.rot90(sinogramm)))
        self.Sinogram_widgett.canvas.axes.imshow(sinogram,cmap='gray')
        
        self.Sinogram_widgett.canvas.draw()   

     
        if (self.comboBox_2.currentText() == 'With Angle 0, 20,40,--'):
    #        sinogram = radon(self.B_P_phantom, theta=theta_a)
            sinogram=radon(self.B_P_phantom,theta_a)

            reconstruction_fbp = iradon(sinogram, theta=theta_a, filter_name = None)

        elif(self.comboBox_2.currentText() == 'No Filter'):
            sinogram = radon(self.B_P_phantom, theta=theta_b)
            reconstruction_fbp = iradon(sinogram, theta = theta_b, filter_name = None)   
        elif (self.comboBox_2.currentText() == 'Roo7 ll box el tany'):
            sinogram = radon(self.B_P_phantom, theta=theta_b)
            reconstruction_fbp = iradon(sinogram, theta=theta_b, filter_name = self.comboBox_3.currentText() )
      

        self.Laminogram_widgett.canvas.axes.clear()
        self.Laminogram_widgett.canvas.axes.imshow(reconstruction_fbp,cmap='gray')
        self.Laminogram_widgett.canvas.draw()    

    def discrete_radon_transform(self,img, theta):
   
        
        img=ImagePil.fromarray(img)
        numAngles = len(theta)
        sinogram = np.zeros((img.size[0],numAngles))

    
        for n in range(numAngles):
            rotImgObj = img.rotate(theta[n], resample=ImagePil.BICUBIC)
            #rotImgObj=self.ImageRotate(img,theta[n])
            sinogram[:,n] = np.sum(rotImgObj, axis=0)
      
        
        return sinogram    
    # def Erode(self,img1):
    #     #Read the image for erosion
    #     #img1=Pil_img.convert(mode='L')
            
    #     img1=np.asarray(img1)
    #     #img1= cv2.imread("/content/wire.tif",0)
    #     #Acquire size of the image
    #     m,n= img1.shape 
    #     #Show the image
    #     plt.imshow(img1, cmap="gray")
    #     # Define the structuring element
    #     # k= 11,15,45 -Different sizes of the structuring element
    #     k=5
    #     #SE= np.ones((k,k), dtype=np.uint8)
    #     SE= np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])

    #     constant= (k-1)//2
    #     #Define new image
    #     imgErode= np.zeros((m,n), dtype=np.uint8)
    #     #Erosion without using inbuilt cv2 function for morphology
    #     for i in range(constant, m-constant):
    #         for j in range(constant,n-constant):
    #             temp= img1[i-constant:i+constant+1, j-constant:j+constant+1]
    #             product= temp*SE
    #             #imgErode[i,j]= np.average(product)
    #             imgErode[i,j]= np.min(product)
                
    #             #imgErode[i,j]= np.min(product)
    #     return imgErode        
              
        #plt.imshow(imgErode,cmap="gray")
        #cv2.imwrite("Eroded3.png", imgErode)
    def Dilate(self,img1):
        #Read the image for dilation
        #img1=Pil_img.convert(mode='L')
            
        img2=np.asarray(img1)
        #img2= cv2.imread("/content/text.tif",0)
        #Acquire size of the image
        p,q= img2.shape
        #Show the image
        plt.imshow(img2, cmap="gray")
        #Define new image to store the pixels of dilated image
        imgDilate= np.zeros((p,q), dtype=np.uint8)
        #Define the structuring element 
        #SED= np.array([[0,1,0], [1,1,1],[0,1,0]])
        SED= np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]],dtype=np.uint8)

        #return cv2.dilate(img2,SED)
        constant1=2
        #Dilation operation without using inbuilt CV2 function
        for i in range(constant1, p-constant1):
            for j in range(constant1,q-constant1):
                temp= img2[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
                product= temp*SED
                imgDilate[i,j]= np.max(product)
        return imgDilate
        
    def erode2(self,image):
        # try:
        padding=1
        #image = self.binarize_this(image)
          #  self.factor=float(self.spinBoxfactor.value())
        image=np.asarray(image)
        self.kernelsize=5
           #if self.kernelsize%2==0 :
              #self.labelerrormask.setText("kernel size must be odd and greater than 1 and Factor cannot be neagtive")
           
        count=1
        for i in range (2,self.kernelsize):
            if(i%2!=0):
                 count+=1
        padding=count
        mag=self.kernelsize*self.kernelsize
        kernel=np.array([[0,1,1,1,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1],[0,1,1,1,0]],dtype=np.uint8)*255
        #print("the sum",)
        #mag will contiam the sum of the elements (ones in the filter)
        #we divid by it to get resamble numbers
        # kernel=kernel*1/mag
        xKern = kernel.shape[0]
        yKern = kernel.shape[1]
        xofimg = image.shape[0]
        yofimg = image.shape[1]
        output = np.zeros((xofimg, yofimg))
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        for y in range(yofimg):
            
            for x in range(xofimg):
                    
                
                    if((kernel * imagePadded[x: x + xKern, y: y + yKern]).sum()==1365525):
                    #    print("output[x, y]",output[x, y])
                        output[x, y]=255
                    else:
                        output[x, y]=0   
        return output    
        #self.theimgff  = Image.fromarray(output)          
    
    def Opening(self,img1):
            #img1=Pil_img.convert(mode='L')
            
            img1=np.asarray(img1)
            #img1= cv2.imread("/content/wire.tif",0)
            #Acquire size of the image
            m,n= img1.shape 
            #Show the image
            plt.imshow(img1, cmap="gray")
            # Define the structuring element
            # k= 11,15,45 -Different sizes of the structuring element
            k=3
            SE= np.ones((k,k), dtype=np.uint8)
            #SE= np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])

            constant= (k-1)//2
            #Define new image
            imgErode= np.zeros((m,n), dtype=np.uint8)
            #Erosion without using inbuilt cv2 function for morphology
            for i in range(constant, m-constant):
                for j in range(constant,n-constant):
                    temp= img1[i-constant:i+constant+1, j-constant:j+constant+1]
                    product= temp*SE
                    imgErode[i,j]= np.min(product)
            p,q= imgErode.shape
        #Show the image
            #plt.imshow(img2, cmap="gray")
            #Define new image to store the pixels of dilated image
            imgDilate= np.zeros((p,q), dtype=np.uint8)
            #Define the structuring element 
            SED= np.array([[0,1,0], [1,1,1],[0,1,0]])
            #SED= np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
            
            constant1=1
            #Dilation operation without using inbuilt CV2 function
            for i in range(constant1, p-constant1):
                for j in range(constant1,q-constant1):
                    temp= imgErode[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
                    product= temp*SED
                    imgDilate[i,j]= np.max(product)
            return imgDilate
            #self.widget_11.canvas.axes.clear()
            #self.widget_11.canvas.axes.imshow(imgDilate,cmap='gray')
            #self.widget_11.canvas.draw()          
    def Closing(self,img1):
        #img1=Pil_img.convert(mode='L')
        
        img2=np.asarray(img1)
        #img2= cv2.imread("/content/text.tif",0)
        #Acquire size of the image
        p,q= img2.shape
        #Show the image
        #plt.imshow(img2, cmap="gray")
        #Define new image to store the pixels of dilated image
        imgDilate= np.zeros((p,q), dtype=np.uint8)
        #Define the structuring element 
        SED= np.array([[0,1,0], [1,1,1],[0,1,0]])
        #SED= np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])

        constant1=1
        #Dilation operation without using inbuilt CV2 function
        for i in range(constant1, p-constant1):
            for j in range(constant1,q-constant1):
                temp= img2[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
                product= temp*SED
                imgDilate[i,j]= np.max(product)
        m,n= imgDilate.shape 
        #Show the image
        #plt.imshow(img1, cmap="gray")
        # Define the structuring element
        # k= 11,15,45 -Different sizes of the structuring element
        k=3
        SE= np.ones((k,k), dtype=np.uint8)
        #SE= np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])

        constant= (k-1)//2
        #Define new image
        imgErode= np.zeros((m,n), dtype=np.uint8)
        #Erosion without using inbuilt cv2 function for morphology
        for i in range(constant, m-constant):
            for j in range(constant,n-constant):
                temp= imgDilate[i-constant:i+constant+1, j-constant:j+constant+1]
                product= temp*SE
                imgErode[i,j]= np.min(product)
        return imgErode
        #self.widget_10.canvas.axes.clear()
        #self.widget_10.canvas.axes.imshow(imgErode,cmap='gray')
        #self.widget_10.canvas.draw()             
    def mono(self,image):
        #erode
        image=image.convert(mode='L')
        erodedimg=self.erode2(image)
        self.widget_8.canvas.axes.clear()
        self.widget_8.canvas.axes.imshow(erodedimg,cmap='gray')
        self.widget_8.canvas.draw()
        dilimg=self.Dilate(image)
        self.widget_9.canvas.axes.clear()
        self.widget_9.canvas.axes.imshow(dilimg,cmap='gray')
        self.widget_9.canvas.draw()
        closed=self.Dilate(image)
        closed= self.erode2(closed)
        self.widget_10.canvas.axes.clear()
        self.widget_10.canvas.axes.imshow(closed,cmap='gray')
        self.widget_10.canvas.draw()
        opened= self.erode2(image)
        opened=self.Dilate(opened)
        
        self.widget_11.canvas.axes.clear()
        self.widget_11.canvas.axes.imshow(opened,cmap='gray')
        self.widget_11.canvas.draw()
        noise=self.Opening(image)
        noise=self.Closing(noise)
        self.widget_12.canvas.axes.clear()
        self.widget_12.canvas.axes.imshow(noise,cmap='gray')
        self.widget_12.canvas.draw()

    def errorMssg(self, txt):
            QMessageBox.critical(self, "Error", txt)        
from pyqtgraph import PlotWidget
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()        