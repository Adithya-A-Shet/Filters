# -*- coding: utf-8 -*-
"""
Created on Sat feb  27 20:04:54 2021


"""

from tkinter import*
from tkinter import messagebox
from tkinter import filedialog
import cv2
from skimage.restoration import denoise_nl_means,estimate_sigma,denoise_tv_chambolle
from skimage import io,img_as_float,img_as_ubyte
from skimage.filters import sobel
import numpy as np
import bm3d


top=Tk()
top.title("Image Processing")
top.resizable(0,0)
top.geometry("750x480+370+150")

top.configure(bg="#d1c4e9")

#img=cv2.imread('C:/Users/91776/OneDrive/Pictures/Camera Roll/WIN_20210222_18_58_24_Pro.jpg',0)

def loc():  #to get the location of image
    global dir # this dir used to provide paths
    dir=filedialog.askopenfilename() #gets location (change the code here this code gets only the directory but we need .jpg file)
    if dir=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        locerror.config(text=dir)
    
def original():
    img=cv2.imread(dir)
    cv2.imshow("Original",img)

def gaussian_fil(): #Gaussian Filter works fine
    #print(dir)
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,0)
        gaussian_img=cv2.GaussianBlur(img,(3,3),0,borderType=cv2.BORDER_CONSTANT)
        cv2.imshow("Gaussian Filter",gaussian_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
def median_fil():  #Median Filter works fine
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,0)
        median_img=cv2.medianBlur(img,3)
        cv2.imshow("Median Filter",median_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
def bilateral_fil():  #Bilateral Filter works fine
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,0)
        bilateral_img=cv2.bilateralFilter(img,5,20,100,borderType=cv2.BORDER_CONSTANT)
        cv2.imshow("Bilateral Filter",bilateral_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def nlm_fil():  #Non-local means Filter works fine
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,1)
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img1=img_as_float(img1)
        sigma_est=np.mean(estimate_sigma(img1,multichannel=True))
        nlm_img=denoise_nl_means(img1,h=1.15*sigma_est,
                                 fast_mode=True,patch_size=5,patch_distance=3,multichannel=True)
        nlm_img=img_as_ubyte(nlm_img)
        nlm_img=cv2.cvtColor(nlm_img,cv2.COLOR_RGB2BGR)
        cv2.imshow("Non-local means Filter",nlm_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def tv_fil():  #Toatl variation Filter works fine
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,1)
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img1=img_as_float(img1)
        tv_img=denoise_tv_chambolle(img1,weight=0.1,eps=0.0002,n_iter_max=200,
                                    multichannel=True)
        tv_img=img_as_ubyte(tv_img)
        tv_img=cv2.cvtColor(tv_img,cv2.COLOR_RGB2BGR)
        cv2.imshow("Toatl variation Filter",tv_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
def bm3d_fil(): #Block Matching & 3D Filter works fine
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,1)
        img1=img_as_float(img)
        bm3d_img=bm3d.bm3d(img1, sigma_psd=0.05,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        cv2.imshow("Block Matching & 3D Filter",bm3d_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def edge_sobel():  #Edge Detection Sobel works fine
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,0)
        sobel_img=sobel(img)
        cv2.imshow("Edge Detection Sobel",sobel_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
def img_fil_fourier(): #Filter using Fourier transform donot work check this
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,0)
        dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift=np.fft.fftshift(dft)
        magnitude_spectrum=20 * np.log((cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
        
        rows,cols=img.shape
        crow,ccol=int(rows/2),int(cols/2)
        mask=np.ones((rows,cols,2),np.uint8)
        r=80
        center=[crow,ccol]
        x,y=np.ogrid[:rows,:cols]
        mask_area=(x-center[0])**2 + (y-center[1])**2 <= r*r
        mask[mask_area]=0
        
        fshift=dft_shift*mask
        fshift_mask_mag=20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
        f_ishift=np.fft.ifftshift(fshift)
        img_back=cv2.idft(f_ishift)
        img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        
        cv2.imshow("Filter using Fourier transform",img_back)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
def clahe_fil(): #Histogram Equilization CLAHE works fine.
    if locerror['text']=="":
        messagebox.showerror("Error","Please choose the locattion")
    else:
        img=cv2.imread(dir,1)
        lab_img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab_img)
        
        clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        clahe_img=clahe.apply(l)
        
        up_image=cv2.merge((clahe_img,a,b))
        
        clahe_img=cv2.cvtColor(up_image,cv2.COLOR_LAB2BGR)
        cv2.imshow("Histogram Equilization CLAHE",clahe_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    
b=Button(top,text="Click here to Choose  location of image which need to be processed",
         command=loc,fg="floralwhite",bg="#926fce",font=("times new roman",14,"bold"),
         bd=5,relief="groove")
b.place(x=90,y=25)

locerror=Label(top,text="",bg="#d1c4e9",fg="floralwhite",
               font=("times new roman",15,"bold"))
locerror.place(x=80,y=100)

tot=Button(top,text="Gaussian Filter",command=gaussian_fil,fg="floralwhite",
           bg="#512da8",font=("times new roman",15,"bold"),
           bd=5).place(x=100,y=160)
tot=Button(top,text="  Median Filter  ",command=median_fil,fg="floralwhite",
           bg="#512da8",font=("times new roman",15,"bold"),bd=5).place(x=280,y=160)
tot=Button(top,text="  Bilateral Filter  ",command=bilateral_fil,fg="floralwhite",
           bg="#512da8",font=("times new roman",15,"bold"),bd=5).place(x=465,y=160)
tot=Button(top,text="  Non-local means Filter  ",command=nlm_fil,fg="floralwhite",bg="#512da8",
           font=("times new roman",15,"bold"),bd=5).place(x=100,y=240)
tot=Button(top,text="  Filter using Fourier transform  ",command=img_fil_fourier,fg="floralwhite",bg="#512da8",
           font=("times new roman",15,"bold"),bd=5).place(x=340,y=240)
tot=Button(top,text="  Total variation Filter  ",command=tv_fil,fg="floralwhite",bg="#512da8",
           font=("times new roman",15,"bold"),bd=5).place(x=410,y=320)
tot=Button(top,text="  Block Matching & 3D Filter  ",command=bm3d_fil,fg="floralwhite",
           bg="#512da8",font=("times new roman",15,"bold"),bd=5).place(x=100,y=320)
tot=Button(top,text="  Edge Detection Sobel   ",command=edge_sobel,fg="floralwhite",bg="#512da8",
           font=("times new roman",15,"bold"),bd=5).place(x=410,y=400)
tot=Button(top,text="  Histogram Equilization CLAHE   ",command=clahe_fil,fg="floralwhite",bg="#512da8",
           font=("times new roman",15,"bold"),bd=5).place(x=90,y=400)
tot=Button(top,text="  Original   ",command=original,fg="floralwhite",bg="#512da8",
           font=("times new roman",15,"bold"),bd=5).place(x=515,y=90)

top.mainloop()