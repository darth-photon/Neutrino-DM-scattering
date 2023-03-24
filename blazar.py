import numpy as np
import tables
import time
import csv
import os
from playsound import playsound
from threading import Thread
# import winsound
# import sys
# import cv2
# from ffpyplayer.player import MediaPlayer

from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirectedo
from termcolor import cprint 
from pyfiglet import figlet_format
import pyfiglet

import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad

from numpy import linalg as LA
from sympy import Matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import multiprocessing as mp


tobs = 898*24*3600
F0 = 13.22
F1 = 1.498
F2 = -0.00167
F3 = 4.119

# data = pd.read_fwf("IceCube_data_from_2008_to_2017_related_to_analysis_of_TXS_0506+056/Aeff_IC86c.txt", header=None)
# data.drop(index=0,inplace=True)
# data.columns = ['a','b','c']
# data = data.astype(float)
# data[['a','b']] = pow(10,data[['a','b']])/1000
# func = interp1d(data['a'],data['c'], fill_value="extrapolate", kind='linear')

# def Afunc(x):
#     return func(x)[()]


# Initial flux
def phi(x):
    return 10**(-F0 - (F1*np.log10(x))/(1 + F2 * np.abs(np.log10(x))**F3))

# Function that describes the effective area of interaction in experiments like ICECUBE etc
def Afunc(x):
    y = np.log10(x)
    return 10**(3.57 + 2.007*y -0.5263* y**2 +0.0922 * y**3 -0.0072* y**4)

# Function that calculates the Right Hand Side of the Cascade equation 
def RHS_matrices(energy_nodes, dxs_array, ReverseTime=False):
    NumNodes = len(energy_nodes)
    DeltaE = np.diff(np.log(energy_nodes))
    RHSMatrix = np.zeros((NumNodes, NumNodes))
    # fill in diagonal terms
    for i in range(NumNodes):  #E_out
        for j in range(i + 1, NumNodes):  #E_in
            RHSMatrix[i][j] = DeltaE[j - 1] * dxs_array[j][i] * energy_nodes[j]**-1
    return RHSMatrix

# Function that vectorizes the cascade equation and calculates the eigenvectors and eigenvalues
def eigcalc(num,a,b):
    logemin = np.log10(290)
    logemax = np.log10(10**4)
    energy_nodes = np.logspace(logemin, logemax, num)
    #phi_0
    phi_0 = 10**(-F0-(F1*np.log10(energy_nodes))/(1 + F2 * np.abs(np.log10(energy_nodes))**F3)) 
    #cross section matrix
    sigma_array =  (a/b)*(1 - 1/(2*b*energy_nodes) * np.log(1 + 2*b* energy_nodes))
    #differential cross section matrix
    dxs_array  = np.empty([num,num])
    for i in range(num):
        for j in range(num): 
            dxs_array[i][j] = quad(lambda x: a*(energy_nodes[i]/x)* 1/((1 + 2*b*(x-energy_nodes[i]))**2),energy_nodes[i],10**4)[0]
    
    #getting the RHS matrix using cas.py
    RHN = RHS_matrices(energy_nodes, dxs_array, ReverseTime=False)
    
    #calculating eigenvalues, eigenvectors and solving for the coefficients
    w, v = LA.eig((-np.diag(sigma_array) + RHN))
    ci = LA.solve(v, phi_0)
    return w, v, ci, energy_nodes

# Intermediate function to evaluate the attenuated flux at required energies
def phifunc(E,num,a,b):
    w, v, ci, energy_nodes = eigcalc(num,a,b)
    phisol = np.dot(v, (ci*np.exp(w)))
    # phisol = phi_0 * np.exp(w * a) * energy_nodes**(-2)
    phisolinterp = interp1d(energy_nodes, phisol)
    # print( np.dot(v, ci)*np.exp(w * a) * energy_nodes**(-2) )
    return phisolinterp(E)

# Calculates events for a range of A and B values (Cross Section (CS) and Differential CS should be parameterized using A and B)
# Call the function "plottingAvsB" to plot the A vs B allowed parameter space 
def events(N,Neig):
    Aval = np.logspace(-3,0.0,num=N,endpoint=True) 
    Bval = np.logspace(-6,0,num=N,endpoint=True)
    dat_fin = np.zeros([N*N,3])

    steps = 5000
    deltaE = (10**4-10**(np.log10(290)))/steps
    enn = np.linspace(10**(np.log10(290)),10**4,steps)

    start_time = time.time()
    if os.path.exists("events.csv"):
        os.remove("events.csv")
    # thread = Thread(target=audio)
    # thread.start()
    # winsound.PlaySound("titanic.mp3", winsound.SND_ALIAS|winsound.SND_ASYNC)
    # playsound('MI.mp3',False)
    # time.sleep(0.2)
    with open('events.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(N):
            for j in range(N):
                tmp=0.0
                tmp = np.sum(tobs* phifunc(enn,Neig, Aval[i],Bval[j])*Afunc(enn))*deltaE
                print(i,j)
                data = [Aval[i], Bval[j], tmp]
                writer.writerow(data)
    end_time = time.time()
    print("Time taken: ", end_time - start_time,"\n")
    videoplay()
    plottingAvsB()

def videoplay():
    # videoName = 'rick.mp4'

    # #create a videoCapture Object (this allow to read frames one by one)
    # video = cv2.VideoCapture(videoName)
    # #check it's ok
    # if video.isOpened():
    #     print('Video Succefully opened')
    # else:
    #     print('Something went wrong check if the video name and path is correct')


    # #define a scale lvl for visualization
    # scaleLevel = 3 #it means reduce the size to 2**(scaleLevel-1)


    # windowName = 'Video Reproducer'
    # cv2.namedWindow(windowName )
    # #let's reproduce the video
    # while True:
    #     ret,frame = video.read() #read a single frame 
    #     if not ret: #this mean it could not read the frame 
    #         print("Could not read the frame")   
    #         cv2.destroyWindow(windowName)
    #         break

    #     reescaled_frame  = frame
    #     for i in range(scaleLevel-1):
    #         reescaled_frame = cv2.pyrDown(reescaled_frame)

    #     cv2.imshow(windowName, reescaled_frame )

    #     waitKey = (cv2.waitKey(1) & 0xFF)
    #     if  waitKey == ord('q'): #if Q pressed you could do something else with other keypress
    #         print("closing video and exiting")
    #         cv2.destroyWindow(windowName)
    #         video.release()
    #         break
    video_path="rick.mp4"
    def PlayVideo(video_path):
        video=cv2.VideoCapture(video_path)
        player = MediaPlayer(video_path)
        while True:
            grabbed, frame=video.read()
            audio_frame, val = player.get_frame()
            if not grabbed:
                print("End of video")
                break
            if cv2.waitKey(28) & 0xFF == ord("q"):
                break
            cv2.imshow("Video", frame)
            if val != 'eof' and audio_frame is not None:
                #audio
                img, t = audio_frame
        video.release()
        cv2.destroyAllWindows()
    PlayVideo(video_path)

# plots the allowed parameter space of A and B and extracts the plot points A and B
# to evaluate the required new physics parameters such as coupling and masses of new particles

def plottingAvsB():
    dat_fin = np.loadtxt("events.csv", delimiter=",")
    df = pd.DataFrame(dat_fin, columns = ['Column_A','Column_B','Column_C'])
    xcol, ycol, zcol = 'Column_A', 'Column_B', 'Column_C'
    df = df.sort_values(by=[xcol, ycol])

    xvals = df[xcol].unique()
    yvals = df[ycol].unique()
    zvals = df[zcol].values.reshape(len(xvals), len(yvals)).T

    mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'

    plt.rcParams['axes.linewidth'] = 2
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['axes.linewidth'] = 2

    fig = plt.figure(figsize=(10,8))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    ax = fig.add_subplot(111)
    ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)
        
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xscale('log')
    plt.yscale('log')

    CP = plt.contour(xvals, yvals, zvals, levels=[0.1], colors='r',linestyles='solid')
    # plt.tricontourf(xvals, yvals, zvals, levels=[0.1, 0.5], colors='r',linestyles='solid')

    x_coords = CP.allsegs[0][0][:, 0]
    y_coords = CP.allsegs[0][0][:, 1]

    plt.ylabel(r'$B$',fontsize=22)
    plt.xlabel(r'$A$',fontsize=22)

    ax.set_xlim([10**-3, 10**-0.0])
    ax.set_ylim([1e-6, 1e0])
    if os.path.exists("AvsB_blazar.pdf"):
        os.remove("AvsB_blazar.pdf")
    # ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    # ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.savefig('AvsB_blazar.pdf')
    # plt.show()
    
    plotmvsg(x_coords,y_coords)

# Converts the A and B values to the new physics parameters (depends on the parameterization of CS)

def plotmvsg(x_coords,y_coords):
    text = pyfiglet.figlet_format("DM - Neutrino interaction")
    print(text)
    print("*****************************************************************************\n")
    print("The available dark matter models are: \n 1)BM1, BM1` \n 2)BM2, BM2` \n 3)BM3, BM3`\n")
    print("*****************************************************************************\n")
    model = input("Choose the dark matter model: ")
    dmmass = float(input("\nEnter the dark matter mass in MeV: "))
    models = [["BM1",31.4,1.0],["BM1p",31.9,1.0],["BM2",30.0,0.48],["BM2p",30.8,0.73],["BM3",28.7,0.43],["BM3p",29.5,0.66]]

    [(A := models[i][1], B:=models[i][2]) for i in range(len(models)) if model==models[i][0]]
    A = float(A)
    B = float(B)
    SigmaChi = 10**A * (dmmass)**(1-B) * (1.98* 10**-14)**2 * 10**-3
    m_nodes = np.linspace(10**-3,10**3,20)
    mvsg = np.empty([len(x_coords),2])
    for i in range(len(x_coords)):
        mvsg[i,0] = np.sqrt(dmmass/(y_coords[i]))
        mvsg[i,1] = mvsg[i,0]**2 * np.sqrt(x_coords[i]/(SigmaChi * 10**3)) * np.sqrt(4*np.pi)
    
    # mvsg_func = interp1d(mvsg[i,0],mvsg[i,1])
    # mvsg_dat = [(m_nodes[i],mvsg_func(m_nodes[i])) for i in range(20)]

    mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
    plt.rcParams['axes.linewidth'] = 2
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['axes.linewidth'] = 2

    fig = plt.figure(figsize=(10,8))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    ax = fig.add_subplot(111)
    ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)
        
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xscale('log')
    plt.yscale('log')

    plt.plot(mvsg[2::,0], np.sqrt(mvsg[2::,1]))

    plt.ylabel(r"$g_{\nu}$",fontsize=22)
    plt.xlabel(r"$m_{Z'}\mathrm{~(GeV)}$",fontsize=22)

    # ax.set_xlim([1e-1,1e+5])
    # ax.set_ylim([1e-33,1e-26])

    # ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    # ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    if os.path.exists("mvsg_blazar_mchi="+str(dmmass)+"-"+str(model)+".pdf"):
        os.remove("mvsg_blazar_mchi="+str(dmmass)+"-"+str(model)+".pdf")
    plt.savefig("mvsg_blazar_mchi="+str(dmmass)+"-"+str(model)+".pdf")
    print("Thy Bidding is done, My Master \n")
    # playsound('good.mp3')
    # plt.show()
# print("////////////////////////")
# print("Dark Matter - Neutrino Scattering in Blazars")
# print("////////////////////////")
def audio():
    # playsound('good.mp3')
    # playsound('laughing.mp3')
    playsound('MI.mp3')
    # playsound('blaster.mp3')

text = pyfiglet.figlet_format("Blazar", font="starwars")
print(text)
# cprint(figlet_format('DM - Neutrino scattering in Blazars', font='starwars'))


N = int(input("Enter the no of A, B splits: "))
Neig = int(input("Enter the no of eigenvalue splits: "))
events(N, Neig)