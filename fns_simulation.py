#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: D.Brueckner

"""

import numpy as np
import fns_data_wrapper
import copy

def simulate(params,modes):
    
    
    (mode_confined,Nx,Ny,N_dim,N_t_max,t_step_record,dt,N_samples) = modes
    (y_boundary,n_boundary,y_flat,k_elastic,l,beta_vec,sigma,sigma_alpha,lambda_stress_pol,beta_dxxp_vec,alpha,lambda_CIL,l_thresh,x_obstacle,stiffness_obstacle) = params
    x_width = 2*(y_flat + y_boundary)

    lambda_stress_pol_vec = lambda_stress_pol*np.array([1,1]) 

    delta_t = dt*t_step_record
    N_t_record = int(N_t_max/t_step_record)
    sqrt_dt = np.sqrt(delta_t)

    c_list = calc_clist(Nx,Ny)
    list_template = calc_clist(Nx,Ny)

    X_record = np.zeros((Nx,Ny,N_dim,N_t_record,N_samples))
    P_record = np.zeros((Nx,Ny,N_dim,N_t_record,N_samples))
    
    N_break_record = np.zeros(N_samples)
    
    for it in range(0,N_samples):
        
        print(it)
        
        force_node = np.zeros((Nx,Ny,N_dim,N_t_max))
        CIL_vector_node = np.zeros((Nx,Ny,N_dim,N_t_max))
        align_node_dxxp = np.zeros((Nx,Ny,N_dim,N_t_max))
        force_bond_list_record = []

        c_list_record = []
        
        X = np.zeros((Nx,Ny,N_dim,N_t_max))
        P = np.zeros((Nx,Ny,N_dim,N_t_max))
        V = np.zeros((Nx,Ny,N_dim,N_t_max))
        
        if sigma_alpha == 0:
            alpha_all = np.ones((Nx,Ny))
        else:
            alpha_all = 1+sigma_alpha*np.random.randn(Nx,Ny)

        #INITIALIZE
        for k in range(0,Ny):
            X[0,k,0,0] = (1-np.mod(k,2))*0.5
            for j in range(0,Nx):
                if j>0:
                    X[j,k,0,0] = X[j-1,k,0,0] + 1
                
                X[j,k,1,0] = k-(Ny-1)/2

                #random initial polarities
                P[j,k,0,0] = 0.3*np.random.normal()
                P[j,k,1,0] = 0.3*np.random.normal()
        
        count_record = 0
        
        list_copy = copy.deepcopy(c_list)
        c_list_record.append(list_copy)
        for t in range(0,N_t_max-1):
            
            if Nx>1:
                #BREAK BONDS BEYOND THRESHOLD
                for k in range(0,Ny):
                    for j in range(0,Nx):     
                        
                        N_connections = len(c_list[k][j])
                        if len(c_list[k][j][0])==0:
                            N_connections = 0
         
                        b = 0
                        while b < N_connections:
                            n_vec = X[c_list[k][j][b][1],c_list[k][j][b][0],:,t] - X[j,k,:,t] #vector connecting j and it
                            n_norm = np.linalg.norm(n_vec)
    
                            if n_norm > l_thresh:
                                N_break_record[it] += 1
                                if N_connections>1:
                                    del c_list[k][j][b]
                                    N_connections-=1
                                else:
                                    c_list[k][j][b] = []
                                    N_connections-=1     
                            b += 1  
    
                force_node[:,:,:,t],force_bond_list = calc_force_node(k_elastic,l,c_list,list_template,Nx,Ny,N_dim,X[:,:,:,t])
                align_node_dxxp[:,:,:,t] = calc_align_node_dev(beta_dxxp_vec,c_list,Nx,Ny,N_dim,P[:,:,:,t])
                CIL_vector_node[:,:,:,t] = calc_CIL_node(c_list,Nx,Ny,N_dim,X[:,:,:,t])
        
            for k in range(0,Ny):

                for j in range(0,Nx):
                    if mode_confined == 1: #confined to 1D channel
                        F_boundaries = F_y( X[j,k,:,t],y_boundary,n_boundary,Ny,y_flat,x_width,mode_confined )
                    elif mode_confined == 2: #obstacle (blunt end)
                        y_boundary_obstacle = 1
                        F_boundaries = F_y( X[j,k,:,t],y_boundary,n_boundary,Ny,y_flat,x_width,mode_confined ) + F_obstacle(X[j,k,:,t],x_obstacle,y_boundary_obstacle,stiffness_obstacle,n_boundary)  + F_obstacle(X[j,k,:,t],-x_obstacle,y_boundary_obstacle,stiffness_obstacle,n_boundary)
                    else: #unconfined
                        F_boundaries = np.array([0,0])
                        
                    if Nx == 1: #equations for single cell
                        V[j,k,:,t+1] = alpha_all[j,k]*P[j,k,:,t] + F_boundaries
                        X[j,k,:,t+1] = X[j,k,:,t] + dt*V[j,k,:,t+1]  
                        P[j,k,:,t+1] = P[j,k,:,t] + dt*( F_p(P[j,k,:,t]) + np.multiply(beta_vec,V[j,k,:,t])) + sqrt_dt*sigma*np.random.normal(size=(1,N_dim))
                    else: #equation for cell train/cluster
                        V[j,k,:,t+1] = (force_node[j,k,:,t] + alpha_all[j,k]*P[j,k,:,t] + F_boundaries)
                        X[j,k,:,t+1] = X[j,k,:,t] + dt*V[j,k,:,t+1]  
                        
                        SPC_force = force_node[j,k,:,t]
                        P[j,k,:,t+1] = P[j,k,:,t] + dt*( F_p(P[j,k,:,t]) + np.multiply(beta_vec,V[j,k,:,t]) + align_node_dxxp[j,k,:,t] - lambda_CIL*CIL_vector_node[j,k,:,t] - np.multiply(lambda_stress_pol_vec,SPC_force) ) + sqrt_dt*sigma*np.random.normal(size=(1,N_dim))
        
            if(np.mod(t,t_step_record)==0):
                X_record[:,:,:,count_record,it] = X[:,:,:,t]
                P_record[:,:,:,count_record,it] = P[:,:,:,t]
                
                if Nx>1:
                    force_bond_list_record.append(force_bond_list)
                    
                    list_copy = copy.deepcopy(c_list)
                    c_list_record.append(list_copy)
                
                count_record += 1

    data = fns_data_wrapper.StochasticTrajectoryData(delta_t,c_list,X_record,P_record,force_bond_list=force_bond_list_record,c_list_record=c_list_record,N_break_record=N_break_record)



    return data
    

def F_p(p): return p*(1-np.dot(p,p))
def force(a): return a*(1-1/np.linalg.norm(a))
def confinement_potential(y,y_boundary,n): return -n*(y**(n-1)/y_boundary**n)

def F_y(X,y_boundary,n,Ny,y_flat,x_width,mode_confined):
    if(Ny==1):
        result = confinement_potential(X[1],y_boundary,n)*np.array([0,1])
    else: 
        if(np.abs(X[1]) <= y_flat):
            result = np.array([0,0])
        elif(X[1] > y_flat):
            result = confinement_potential(X[1]-y_flat,y_boundary,n)*np.array([0,1])
        elif(X[1] < -y_flat):
            result = confinement_potential(X[1]+y_flat,y_boundary,n)*np.array([0,1])
    return result

def F_obstacle(X,x_obstacle,y_boundary,stiffness,n):
    if x_obstacle<0:
        condition = X[0] >= x_obstacle
    else:
        condition = X[0] <= x_obstacle
    if(condition):
        result = np.array([0,0])
    else:
        result = stiffness*confinement_potential(X[0]-x_obstacle,y_boundary,n)*np.array([1,0])
    return result


def calc_force_node(k_elastic,l,c_list,list_template,Nx,Ny,N_dim,X_here):
    force_node_here = np.zeros((Nx,Ny,N_dim))
    force_bond_list = list_template
    for k in range(0,Ny):
        for j in range(0,Nx):     
            N_connections = len(c_list[k][j])
            if len(c_list[k][j][0])==0:
                N_connections = 0
            sum_force = 0
            for it in range(0,N_connections):
                n_vec = X_here[c_list[k][j][it][1],c_list[k][j][it][0],:] - X_here[j,k,:] #vector connecting j and it

                n_norm = np.linalg.norm(n_vec) #norm of connecting vector
                n_hat = n_vec/n_norm #connecting unit vector
                tension = k_elastic*(n_norm - l) #tension in link (+ for tensile, - for compressive)
                force_bond_list[k][j][it][0] = np.abs(n_hat[0])*tension #x-component of tension (irrespective of direction of n, sign dictated by tension)
                force_bond_list[k][j][it][1] = np.abs(n_hat[1])*tension
                
                sum_force += k_elastic*force(n_vec) #total force acting on j

            force_node_here[j,k,:] = sum_force
    return force_node_here,force_bond_list


def calc_CIL_node(c_list,Nx,Ny,N_dim,X_here):
    CIL_vector_node_here = np.zeros((Nx,Ny,N_dim))
    for k in range(0,Ny):
        for j in range(0,Nx):     
            N_connections = len(c_list[k][j])
            if len(c_list[k][j][0])==0:
                N_connections = 0
            sum_vectors = 0
            for it in range(0,N_connections):
                vec_ij = X_here[c_list[k][j][it][1],c_list[k][j][it][0],:] - X_here[j,k,:] #vector connecting cell to its neighbours
                vec_ij_norm = vec_ij / np.linalg.norm(vec_ij)
                sum_vectors += vec_ij_norm
            CIL_vector_node_here[j,k,:] = sum_vectors
    return CIL_vector_node_here

def calc_align_node_dev(beta_vec,c_list,Nx,Ny,N_dim,align_field):
    align_node_here = np.zeros((Nx,Ny,N_dim))
    for k in range(0,Ny):
        for j in range(0,Nx):     
            N_connections = len(c_list[k][j])
            if len(c_list[k][j][0])==0:
                N_connections = 0
            sum_align = 0
            for it in range(0,N_connections):
                sum_align += align_field[c_list[k][j][it][1],c_list[k][j][it][0],:]
            
            if N_connections > 0:
                align_node_here[j,k,:] = np.multiply(beta_vec, (sum_align/N_connections) - align_field[j,k,:] ) #take deviation from own velocity as alignment driving force
    return align_node_here

def calc_clist(Nx,Ny):
    c_list = []
    for k in range(0,Ny):
        c_list.append([])
        if(Ny==1):
            k=0
            c_list[k].append([[k,1]])
            for j in range(1,Nx-1):     
                c_list[k].append([[k,j-1],[k,j+1]])
            c_list[k].append([[k,Nx-2]])
            
        else:
            if(k==0):
                c_list[k].append([[k,1],[k+1,0],[k+1,1]])
                for j in range(1,Nx-1):     
                    c_list[k].append([[k,j-1],[k,j+1],[k+1,j],[k+1,j+1]])
                c_list[k].append([[k,Nx-2],[k+1,Nx-1]])
                
            elif(k==Ny-1):
                if(k%2 == 0): #EVEN TOP LAYER
                    c_list[k].append([[k,1],[k-1,0],[k-1,1]])
                    for j in range(1,Nx-1):     
                        c_list[k].append([[k,j-1],[k,j+1],[k-1,j],[k-1,j+1]])
                    c_list[k].append([[k,Nx-2],[k-1,Nx-1]])
                else: #ODD TOP LAYER
                    c_list[k].append([[k,1],[k-1,0]])
                    for j in range(1,Nx-1):     
                        c_list[k].append([[k,j-1],[k,j+1],[k-1,j-1],[k-1,j]])
                    c_list[k].append([[k,Nx-2],[k-1,Nx-2],[k-1,Nx-1]])
            else:
                if(k%2 == 0): #EVEN LAYER
                    c_list[k].append([[k+1,0],[k+1,1],[k,1],[k-1,1],[k-1,0]])
                    for j in range(1,Nx-1):     
                        c_list[k].append([[k+1,j],[k+1,j+1],[k,j+1],[k-1,j+1],[k-1,j],[k,j-1]])
                    c_list[k].append([[k,Nx-2],[k+1,Nx-1],[k-1,Nx-1]])
                else: #ODD LAYER
                    c_list[k].append([[k,1],[k+1,0],[k-1,0]])
                    for j in range(1,Nx-1):     
                        c_list[k].append([[k,j-1],[k+1,j-1],[k+1,j],[k,j+1],[k-1,j],[k-1,j-1]]) 
                    c_list[k].append([[k,Nx-2],[k+1,Nx-2],[k+1,Nx-1],[k-1,Nx-1],[k-1,Nx-2]])
    return c_list


def calc_clist_zeros(Nx,Ny):
    c_list_zeros = calc_clist(Nx,Ny)
    for k in range(0,Ny):
        for j in range(0,Nx):     
            N_connections = len(c_list_zeros[k][j])
            for it in range(0,N_connections):
                for d in range(0,2):
                    c_list_zeros[k][j][it][d] = 0.
    
    return c_list_zeros

    
    
