import numpy as np
import tables
import time
import csv
import os

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

class ces():
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

    if os.path.exists("input/Eff_area.csv"):
        def Afunc(x):
            data = np.load("input/Eff_area.csv")
            data = data[1:,:]
            Afunc = interp1d(data[:,0],data[:,2], fill_value="extrapolate", kind='linear')
            return Afunc(x)
    else:
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
    def eigcalc(num, a, b):
        E = np.logspace(np.log10(290), np.log10(10**4), num)

        if os.path.exists("input/flux.csv"):
            phi_0 = np.load("input/flux.csv")
        else:
            #phi_0 - initial unattenuated flux
            phi_0 = 10**(-F0-(F1*np.log10(E))/(1 + F2 * np.abs(np.log10(E))**F3)) 

        if os.path.exists("input/cross_section.csv"):
            sigma_array = np.load("input/cross_section.csv")
        else:
            #cross section matrix
            # eqn = "(a/b)*(1 - 1/(2*b*E) * np.log(1 + 2*b* E))"
            sigma_array = eval(eqn)
            # sigma_array =  (a/b)*(1 - 1/(2*b*energy_nodes) * np.log(1 + 2*b* energy_nodes))
        
        if os.path.exists("input/diff_cross_section.csv"):
            dxs_array = np.load("input/diff_cross_section.csv")
        else:
            #differential cross section matrix
            dxs_array  = np.empty([num,num])
            for i in range(num):
                for j in range(num): 
                    dxs_array[i][j] = quad(lambda x: a*(E[i]/x)* 1/((1 + 2*b*(x-E[i]))**2),E[i],10**4)[0]
        
        #getting the RHS matrix using cas.py
        RHN = RHS_matrices(E, dxs_array, ReverseTime=False)
        
        #calculating eigenvalues, eigenvectors and solving for the coefficients
        w, v = LA.eig((-np.diag(sigma_array) + RHN))
        ci = LA.solve(v, phi_0)
        return w, v, ci, E

    # Intermediate function to evaluate the attenuated flux at required energies
    def phifunc(E,num,a,b):
        w, v, ci, energy_nodes = eigcalc(num,a,b)
        phisol = np.dot(v, (ci*np.exp(w)))
        phisolinterp = interp1d(energy_nodes, phisol)
        return phisolinterp(E)

    # Calculates events for a range of A and B values (Cross Section (CS) and Differential CS should be parameterized using A and B)
    # Call the function "plottingAvsB" to plot the A vs B allowed parameter space 

    def events(Emin, Emax):
        N = 15
        Aval = np.logspace(-3,0.0,num=N,endpoint=True) 
        Bval = np.logspace(-6,0,num=N,endpoint=True)
        dat_fin = np.zeros([N*N,3])

        steps = 5000
        deltaE = (10**np.log10(Emax)-10**np.log10(Emin))/steps
        enn = np.linspace(10**np.log10(Emin),10**np.log10(Emax),steps)

        start_time = time.time()
        if os.path.exists("events_data/events.csv"):
            os.remove("events_data/events.csv")

        with open('events_data/events.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(N):
                for j in range(N):
                    tmp=0.0
                    tmp = np.sum(tobs* phifunc(enn,15, Aval[i],Bval[j])*Afunc(enn))*deltaE
                    print("\rCalculating" + "." * j, end="")
                    data = [Aval[i], Bval[j], tmp]
                    writer.writerow(data)
        end_time = time.time()
        print("\nTime taken: ", end_time - start_time,"\n")
        plot()

    # plots the allowed parameter space of A and B and extracts the plot points A and B
    # to evaluate the required new physics parameters such as coupling and masses of new particles
    def plot():
        inpt = input("Which plot do you want (AvsB, NP, None)? ")
        if inpt=='None':
            print("Alright. Understandable, Have a nice day")
        dat_fin = np.loadtxt("events_data/events.csv", delimiter=",")
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

        if inpt=='AvsB':
            if os.path.exists("plots/AvsB_blazar.pdf"):
                os.remove("plots/AvsB_blazar.pdf")
            plt.savefig('plots/AvsB_blazar.pdf')
        if inpt=='NP':
            plotmvsg(x_coords,y_coords)

    # Converts the A and B values to the new physics parameters (depends on the parameterization of CS)
    def NPparameters(x_coords,y_coords):
        text = pyfiglet.figlet_format("DM matter models")
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
        return mvsg, model, dmmass

    def plotmvsg(x_coords,y_coords):
        mvsg, model, dmmass = NPparameters(x_coords,y_coords)
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

        if os.path.exists("plots/mvsg_blazar_mchi="+str(dmmass)+"-"+str(model)+".pdf"):
            os.remove("plots/mvsg_blazar_mchi="+str(dmmass)+"-"+str(model)+".pdf")
        plt.savefig("plots/mvsg_blazar_mchi="+str(dmmass)+"-"+str(model)+".pdf")
        print("Thy Bidding is done, My Master \n")

    text = pyfiglet.figlet_format("CASCADE SOLVER", font="starwars")
    print(text)

    print("*******************************************")
    print("Dark Matter - Neutrino Scattering in Blazars\n")
    print("Author: シヴァサンカール ShivaSankar K.A \n")
    print("Affiliation: "+u"素粒子物理、北海道大学\n")
    print("Email: shivasankar.ka@gmail.com")
    print("*******************************************")

    # Emin = float(input("Enter the min energy (TeV): "))
    # Emax = float(input("Enter the max energy (TeV): "))
    # eqn = input("Enter the expression for cross section:")
    
    
    
class Model():
    def __init__(self,name, path="./"):
        self.model_name = name
        self.modelpath = path

    def add_input(self,flux,cross_section,diff_cross_section):
        if flux==True:
            if os.path.exists(self.modelpath+"input/flux.csv"):
                self.phi_0 = np.load(self.modelpath+"input/flux.csv")
                print("Flux intialized")
            else:
                print("Flux file not found")
        else:
            self.flux_eqn = flux
            print("Flux intialized: "+self.flux_eqn)

        if cross_section==True:
            if os.path.exists(self.modelpath+"input/cross_section.csv"):
                self.phi_0 = np.load(self.modelpath+"input/cross_section.csv")
                print("Cross section Intialized")
            else:
                print("Cross section file not found")
        else:
            self.xs_eqn = cross_section
            print("Cross section intialized: "+self.xs_eqn)

        if diff_cross_section==True:
            if os.path.exists(self.modelpath+"input/diff_cross_section.csv"):
                self.phi_0 = np.load(self.modelpath+"input/diff_cross_section.csv")
                print("Differential cross section intialized")
            else:
                print("File not Found")
        else:
            self.dxs_eqn = diff_cross_section
            print("Flux Intialized: "+self.dxs_eqn)
    print("done")
    # events(Emin, Emax)