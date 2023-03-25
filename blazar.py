import numpy as np
import tables
import time
import csv
import os
import warnings

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

text = pyfiglet.figlet_format("CASCADE SOLVER", font="starwars")
print(text)

print("*******************************************")
print("Dark Matter - Neutrino Scattering in Blazars\n")
print("Author: シヴァサンカール ShivaSankar K.A \n")
print("Affiliation: "+u"素粒子物理、北海道大学\n")
print("Email: shivasankar.ka@gmail.com")
print("*******************************************")

class ces():
    def set_model(self,model):
        self.model = model  

    def flux(self,E):
        model = self.model
        return eval(model.flux)
    
    def sigma_array(self, E, a, b):
        model  = self.model
        return eval(model.xs)   
    
    # # Function that calculates the Right Hand Side of the Cascade equation 
    # def RHS_matrices(self,energy_nodes, dxs_array):
    #     self.NumNodes = len(energy_nodes)
    #     self.DeltaE = np.diff(np.log(energy_nodes))
    #     self.RHSMatrix = np.zeros((self.NumNodes, self.NumNodes))
    #     # fill in diagonal terms
    #     for i in range(self.NumNodes):  #E_out
    #         for j in range(i + 1, self.NumNodes):  #E_in
    #             self.RHSMatrix[i][j] = self.DeltaE[j - 1] * dxs_array[j][i] * energy_nodes[j]**-1
    #     return self.RHSMatrix
    
    def dxs_eval(self, E, x, a, b):
        model = self.model
        return eval(model.dxs)
    
    def eff_area(self,E):
        model = self.model
        return eval(model.eff_area)

    # Function that vectorizes the cascade equation and calculates the eigenvectors and eigenvalues
    def eigcalc(self, Energy, num, a, b):
        model  = self.model
        E = np.logspace(np.log10(290), np.log10(10**4), num)

        phi_0 = self.flux(E)
        sigma_array = self.sigma_array(E,a,b)

        dxs_array  = np.empty([num,num])
        for i in range(num):
            # ran = np.linspace(E[i], 10**4, 5000)
            # tot = np.diff(ran)[1] * np.sum([self.dxs_eval(E[i],x,a,b) for x in ran])
            for j in range(num): 
                # dxs_array[i][j] = tot
                dxs_array[i][j] = quad(lambda x: self.dxs_eval(E[i],x,a,b),E[i],10**4)[0]

        #getting the RHS matrix using cas.py
        # RHN = self.RHS_matrices(E, dxs_array)

        DeltaE = np.diff(np.log(E))
        RHN = np.zeros((len(E), len(E)))
        # fill in diagonal terms
        for i in range(len(E)):  #E_out
            for j in range(i + 1, len(E)):  #E_in
                RHN[i][j] = DeltaE[j - 1] * dxs_array[j][i] * E[j]**-1
        
        #calculating eigenvalues, eigenvectors and solving for the coefficients
        w, v = LA.eig((-np.diag(sigma_array) + RHN))
        ci = LA.solve(v, phi_0)
        phisol = np.dot(v, (ci*np.exp(w)))
        phisolinterp = interp1d(E, phisol)
        return phisolinterp(Energy)

    # def attenuated_flux(self,E,num,a,b):
    #     if os.path.exists("events_data/attenuated_flux.csv"):
    #         phisol = np.genfromtxt("events_data/attenuated_flux.csv", delimiter=',')
    #     else:
    #         self.eigcalc(E, num,a,b)
    #         phisol = np.genfromtxt("events_data/attenuated_flux.csv", delimiter=',')
    #     print(E.shape,phisol.shape)
    #     phisolinterp = interp1d(E, phisol)
    #     return phisolinterp(E)
    # Calculates events for a range of A and B values (Cross Section (CS) and Differential CS should be parameterized using A and B)
    # Call the function "plottingAvsB" to plot the A vs B allowed parameter space 

    def events(self,Emin, Emax,t_obs):
        N = 10
        Aval = np.logspace(-3,0.0,num=N,endpoint=True) 
        Bval = np.logspace(-6,0,num=N,endpoint=True)

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
                    tmp = np.sum(t_obs* self.eigcalc(enn,10, Aval[i],Bval[j])*self.eff_area(enn))*deltaE
                    print("\rCalculating" + "." * j, end="")
                    data = [Aval[i], Bval[j], tmp]
                    writer.writerow(data)
        end_time = time.time()
        print("\nTime taken: ", end_time - start_time,"\n")
        self.plot()

    # plots the allowed parameter space of A and B and extracts the plot points A and B
    # to evaluate the required new physics parameters such as coupling and masses of new particles
    def plot(self):
        inpt = input("Which plot do you want (AvsB, NP (NewPhysics), None)? ")
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
            self.plotmvsg(x_coords,y_coords)

    # Converts the A and B values to the new physics parameters (depends on the parameterization of CS)
    def NPparameters(self,x_coords,y_coords):
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

    def plotmvsg(self,x_coords,y_coords):
        mvsg, model, dmmass = self.NPparameters(x_coords,y_coords)
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

    # Emin = float(input("Enter the min energy (TeV): "))
    # Emax = float(input("Enter the max energy (TeV): "))
    # eqn = input("Enter the expression for cross section:")
    
class Model():
    def __init__(self,name, path="./"):
        self.model_name = name
        self.modelpath = path

    def add_input(self,flux,cross_section,diff_cross_section,Effective_area):
        if flux==True:
            if os.path.exists(self.modelpath+"input/flux.csv"):
                self.phi_0 = np.load(self.modelpath+"input/flux.csv")
                print("Flux intialized")
            else:
                warnings.warn("Flux file not found: {}".format(self.modelpath+"input/flux.csv"))
        else:
            self.flux = flux
            print("Flux intialized: "+self.flux)

        if cross_section==True:
            if os.path.exists(self.modelpath+"input/cross_section.csv"):
                self.xs = np.load(self.modelpath+"input/cross_section.csv")
                print("Cross section Intialized")
            else:
                warnings.warn("Cross section file not found: {}".format(self.modelpath+"input/cross_section.csv"))
        else:
            self.xs = cross_section
            print("Cross section intialized: "+self.xs)

        if diff_cross_section==True:
            if os.path.exists(self.modelpath+"input/diff_cross_section.csv"):
                self.dxs = np.load(self.modelpath+"input/diff_cross_section.csv")
                print("Differential cross section intialized")
            else:
                print("File not Found")
        else:
            self.dxs = diff_cross_section
            print("Flux Intialized: "+self.dxs)

        if Effective_area==True:
            if os.path.exists(self.modelpath+"input/Eff_area.csv"):
                self.eff_area = np.load(self.modelpath+"input/Eff_area.csv")
                data = np.load("input/Eff_area.csv")
                print("Differential cross section intialized")
            else:
                print("File not Found")
        else:
            self.eff_area = Effective_area
            print("Effective Area Intialized: "+self.eff_area)