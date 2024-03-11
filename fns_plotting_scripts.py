#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:42:32 2021

@author: D.Brueckner

2D HEXAGONAL lattice, 1 or 2 layers
cleaned up code with node matrices
not yet implemented fully automatic connectivity matrix

"""

import numpy as np
import fns_data_wrapper
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

import dill as pickle


def plot_movie(data,directory,mode_confined,mode_centered,mode_plotting,file_name,Ny,x_obstacle=0):
    
    X_LIM_standard = 15
    
    l=1
    y_boundary = 0.25
    y_flat = (Ny-1)*l/2
    y_line = y_flat + y_boundary
    
    if(mode_confined):
        if mode_centered=='turn90':
            fig_size = [int((x_obstacle+10)/4),int(2*(X_LIM_standard/2+y_line)/4)] 
        elif mode_centered=='opening':
            fig_size = [12,3]
        else:
            fig_size = [12,2] 
    else:
        fig_size = [5,5] 
    
    params = {'figure.figsize': fig_size}
    plt.rcParams.update(params)

    N_t_max_movie = int(data.N_t_max)
    N_pics = int(N_t_max_movie/2)
    N_framerate = int(N_t_max_movie/N_pics)
    
    
    
    
    subdirectory_plot_movie = directory + '/movie' + file_name
    if not os.path.exists(subdirectory_plot_movie):
        os.mkdir(subdirectory_plot_movie)
    
    if mode_plotting == 'MSM':
        stress_max = max(max(max(max(max(data.force_bond_list)))))
        stress_min = min(min(min(min(min(data.force_bond_list)))))
        stress_lim = max([np.abs(stress_max),np.abs(stress_min)])
        norm = mpl.colors.Normalize(vmin=-stress_lim, vmax=stress_lim)
        cmap = mpl.cm.get_cmap('coolwarm')
    
    plt.close('all')
    #https://stackoverflow.com/questions/60620345/create-video-from-python-plots-to-look-like-a-moving-probability-distribution
    fig = plt.figure()
    ax = fig.gca()
    w=0 #vid frame step
    for t in range(0,N_t_max_movie,N_framerate):
        if mode_plotting == 'cells_bonds' or mode_plotting == 'bonds' or mode_plotting == 'MSM':
            for k in range(0,data.Ny):
                for j in range(0,data.Nx):     
                    N_connections = len(data.c_list_record[t][k][j])
                    if len(data.c_list_record[t][k][j][0])==0:
                        N_connections = 0
                    for it in range(0,N_connections):
                        xx = np.zeros((2,2))
                        for d in range(0,data.N_dim):
                            xx[0,d] = data.X[data.c_list_record[t][k][j][it][1],data.c_list_record[t][k][j][it][0],d,t]
                            xx[1,d] = data.X[j,k,d,t]
                        plt.plot(xx[:,0],xx[:,1],color='limegreen',lw=2)
                        if mode_plotting == 'MSM':
                            plt.plot(xx[:,0],xx[:,1],color=cmap(norm(data.force_bond_list[t][k][j][it][0])))
        
        if mode_plotting == 'cells' or mode_plotting == 'cells_bonds':
            if mode_centered == 'pbc':
                for j in range(0,data.Nx):     
                    ax.scatter(data.X[j,0,0,t],data.X[j,0,1,t],s=100,c=j,cmap = 'Greens',vmin=0,vmax=data.Nx)
            else:
                ax.scatter(data.X[:,:,0,t],data.X[:,:,1,t],100, color='limegreen')
            ax.quiver(data.X[:,:,0,t],data.X[:,:,1,t],data.P[:,:,0,t],data.P[:,:,1,t],scale=20,color='fuchsia',width=0.005,headwidth=2)

        
        if mode_plotting == 'polarity':
            ax.quiver(data.X[:,:,0,t],data.X[:,:,1,t],data.P[:,:,0,t],data.P[:,:,1,t],data.P[:,:,0,t],scale=50,cmap='bwr',width=0.005,headwidth=2)
        
        if mode_plotting == 'CIL':
            ax.scatter(data.X[:,:,0,t],data.X[:,:,1,t],100, color='limegreen')
            ax.quiver(data.X[:,:,0,t],data.X[:,:,1,t],data.P[:,:,0,t],data.P[:,:,1,t],scale=20,color='fuchsia',width=0.005,headwidth=2)
            ax.quiver(data.X[:,:,0,t],data.X[:,:,1,t],data.CIL_vector_node[:,:,0,t],data.CIL_vector_node[:,:,1,t],scale=20,color='grey',width=0.005,headwidth=2)
            
        if mode_plotting == 'MSM':
            ax.scatter(data.X[:,:,0,t],data.X[:,:,1,t],100, color='grey')
            ax.quiver(data.X[:,:,0,t],data.X[:,:,1,t],data.P[:,:,0,t],data.P[:,:,1,t],scale=30,color='grey',width=0.005,headwidth=2)
            
            if t==0:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm) #ticks=np.linspace(0,2,N),boundaries=np.arange(-0.05,2.1,.1)
        
        
        
        if(mode_centered==True):
            x_lim = X_LIM_standard
            x_lim_neg = -x_lim
            x_lim_pos = 2.5*x_lim
            """
            #for 1D confluent
            x_lim_neg = 2*x_lim#-x_lim
            x_lim_pos = 3*x_lim
            """
        elif(mode_centered=='pbc'):
            L = data.Nx
            x_lim_neg = -5
            x_lim_pos = L+5
            
            plt.plot([0,0],[-y_line,y_line],color='k')
            plt.plot([L,L],[-y_line,y_line],color='k')
        elif(mode_centered=='obstacle'):
            x_lim = X_LIM_standard
            x_lim_neg = -x_obstacle-5
            x_lim_pos = x_obstacle+5
            
            plt.plot([x_obstacle,x_obstacle],[-y_line,y_line],color='k',lw=2)
            plt.plot([-x_obstacle,-x_obstacle],[-y_line,y_line],color='k',lw=2)
        elif(mode_centered=='opening'):

            x_lim = X_LIM_standard
            x_lim_neg = -x_obstacle/2
            x_lim_pos = 2.5*x_obstacle
        elif(mode_centered=='turn90'):

            x_lim = X_LIM_standard
            x_lim_neg = -5
            x_lim_pos = x_obstacle+5
            
            plt.plot([x_obstacle,x_obstacle],[-1000,y_line],color='k',lw=1)
            plt.plot([x_obstacle-2*y_line,x_obstacle-2*y_line],[-1000,-y_line],color='k',lw=1)
            
            plt.plot([-x_obstacle,-x_obstacle],[-y_line,y_line],color='k',lw=1)
            
            plt.plot([-1000,x_obstacle-2*y_line],[-y_line,-y_line],'-k')
            plt.plot([-1000,x_obstacle],[y_line,y_line],'-k')
            
        else:
            x_lim_neg = min(data.X[:,:,0,:].ravel())
            x_lim_pos = max(data.X[:,:,0,:].ravel())
        
        plt.xlim([x_lim_neg,x_lim_pos])
        
        
        if(mode_confined and not (mode_centered=='turn90' or mode_centered=='obstacle' or mode_centered=='pbc' or mode_centered=='opening')):
            plt.ylim([-2*y_line,2*y_line])
            plt.plot([-1000,1000],[-y_line,-y_line],'-k')
            plt.plot([-1000,1000],[y_line,y_line],'-k')
        elif(mode_centered=='obstacle'):
            plt.ylim([-3*y_line,3*y_line])
            plt.plot([-x_obstacle,x_obstacle],[-y_line,-y_line],'-k',lw=2)
            plt.plot([-x_obstacle,x_obstacle],[y_line,y_line],'-k',lw=2)
        elif(mode_centered=='opening'):
            plt.ylim([-3*y_line,3*y_line])
            plt.plot([-1000,x_obstacle],[-y_line,-y_line],'-k',lw=2)
            plt.plot([-1000,x_obstacle],[y_line,y_line],'-k',lw=2)
        elif mode_centered=='turn90':
            plt.ylim([-x_lim,2*y_line])
        
        plt.axis('off')
        
        plt.savefig(subdirectory_plot_movie + '/' + f'plot_step_{w:04d}.png')
        plt.cla()
        
        w += 1
        
        
    os.chdir(subdirectory_plot_movie)
    print(subdirectory_plot_movie)
    if(os.path.isfile("movie.mp4")):
        os.remove("movie.mp4")
    import subprocess
    subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'plot_step_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'movie.mp4'
        ])
    
    #change back to top directory
    os.chdir('../../../..')
    
    
    return 0



def return_filename(Nx,Ny,mode_confined,lambda_stress_pol,beta,sigma,k_elastic,beta_dxxp,alpha,lambda_CIL,l_thresh):
    paramname = return_paramname(lambda_stress_pol,beta,sigma,k_elastic,beta_dxxp,alpha,lambda_CIL,l_thresh)
    file_name = '_Nx' + str(Nx) + '_Ny' + str(Ny) + '_conf' + str(mode_confined) + paramname
    return file_name

def return_paramname(lambda_stress_pol,beta,sigma,k_elastic,beta_dxxp,alpha,lambda_CIL,l_thresh):
    file_name = '_lambda' + str(np.round(lambda_stress_pol,2)).replace('.','') + '_beta' + str(np.round(beta,2)).replace('.','') + '_betadxxp' + str(np.round(beta_dxxp,2)).replace('.','') + '_lambdaCIL' + str(np.round(lambda_CIL,2)).replace('.','') + '_sigma' + str(np.round(sigma,2)).replace('.','') + '_k' + str(np.round(k_elastic,2)).replace('.','') + '_alpha' + str(np.round(alpha,2)).replace('.','')  + '_lthresh' + str(np.round(l_thresh,2)).replace('.','')  
    return file_name

