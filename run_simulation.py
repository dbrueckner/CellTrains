#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: D.Brueckner

"""

import numpy as np
import os

import fns_simulation as fns_simulation
import fns_plotting_scripts as fns_plotting_scripts
import fns_data_wrapper

N_samples = 1 #number of clusters
N_dim = 2 #number of dimensions
N_t_max = 2000 #max no. of time steps
t_step_record = 10 #sampling of simulation for downstream analysis
dt = 0.01 #time increment

geometry = 'lines' #micropattern geometry

directory = 'data'
if not os.path.exists(directory):
    os.mkdir(directory)

#interaction amplitudes
beta = 1
lambda_CIL = 0.7
beta_dxxp = 1.5
lambda_stress_pol = 0

sigma = 0.05 #noise magnitude
alpha = 1 #scaling of single cell term, fixed to unity
sigma_alpha = 0 #cell-to-cell variability STDEV of single cell term
k_elastic = 40 #adhesion spring constant
l_thresh = 3 #critical extension for cell detaching from cluster
l = 1 #rest length of adhesions

Nx = 8 #length of cluster
Ny = 2 #width of cluster

if geometry == 'lines': #1D line
    x_obstacle = 0
    stiffness_obstacle = 0

    mode_centered = 1
    mode_confined = 1
elif geometry == 'obstacle': #blunt end
    x_obstacle = 20
    stiffness_obstacle = 100
    
    mode_centered = geometry
    mode_confined = 2


#parameters of micropattern geometry
y_boundary = 0.25
y_flat = (Ny-1)*l/2
n_boundary = 2

beta_vec = beta*np.array([1,1]) #aligment interactions in x,y
beta_dxxp_vec = beta_dxxp*np.array([1,1])
modes = (mode_confined,Nx,Ny,N_dim,N_t_max,t_step_record,dt,N_samples) 
params = (y_boundary,n_boundary,y_flat,k_elastic,l,beta_vec,sigma,sigma_alpha,lambda_stress_pol,beta_dxxp_vec,alpha,lambda_CIL,l_thresh,x_obstacle,stiffness_obstacle)

data = fns_simulation.simulate(params,modes)

data_analyzed = fns_data_wrapper.AnalyzeData(data,mode_com=True)

file_name = fns_plotting_scripts.return_filename(Nx,Ny,mode_confined,lambda_stress_pol,beta,sigma,k_elastic,beta_dxxp,alpha,lambda_CIL,l_thresh)

#movie production
mode_plotting = 'cells_bonds'
fns_plotting_scripts.plot_movie(data_analyzed,directory,mode_confined,mode_centered,mode_plotting,file_name,Ny,x_obstacle=x_obstacle)

        