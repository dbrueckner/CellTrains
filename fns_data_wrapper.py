#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:52:27 2020

@author: D.Brueckner
"""

import numpy as np

class StochasticTrajectoryData(object):

    def __init__(self,delta_t,c_list,X,P,force_node=None,align_node=None,align_node_dxxp=None,CIL_vector_node=None,force_bond_list=None,c_list_record=None,N_break_record=None):

        self.Nx = X.shape[0]
        self.Ny = X.shape[1]
        self.N_dim = X.shape[2]
        self.N_t_max = X.shape[3]
        self.N_samples = X.shape[4]
        self.dt = delta_t
        
        self.X = X
        self.P = P
        self.V = np.diff(self.X,axis=3)/self.dt
        
        self.force_node = force_node
        self.align_node_dxxp = align_node_dxxp
        self.align_node = align_node
        self.CIL_vector_node = CIL_vector_node
        self.force_bond_list = force_bond_list
        self.c_list_record = c_list_record
        self.N_break_record=N_break_record
        
        self.c_list = c_list


class AnalyzeData(object):
    
    def __init__(self,data,N_t_final=-1,mode_com=False):
        
        self.Nx = data.Nx
        self.Ny = data.Ny
        self.N_dim = data.N_dim
        self.N_t_max = data.N_t_max
        self.N_samples = data.N_samples
        self.dt = data.dt
        
        self.X = data.X[:,:,:,:,0]
        self.P = data.P[:,:,:,:,0]
        self.V = data.V[:,:,:,:,0]
        self.force_bond_list = data.force_bond_list
        self.c_list_record = data.c_list_record
        self.N_break_record = data.N_break_record
        
        #CoM motion
        if mode_com:
            self.X_com = data.X.mean(axis=(0,1))
            #np.zeros((data.N_dim,data.N_t_max,data.N_samples))

        if N_t_final>0:
            self.X_final = data.X[:,:,:,-N_t_final:,:]
            self.P_final = data.P[:,:,:,-N_t_final:,:]
            self.V_final = data.V[:,:,:,-N_t_final:,:]
        
        #self.force_node = data.force_node[:,:,:,:,0]
        #self.CIL_vector_node = data.CIL_vector_node[:,:,:,:,0]

        self.c_list = data.c_list
        
        self.speed_itav = np.zeros((self.N_samples))
        for it in range(0,self.N_samples):
            self.speed_itav[it] = np.mean(np.abs(np.ravel(data.V[:,:,0,:,it])))
        self.speed_av = np.mean(self.speed_itav)
        self.speed_var = np.var(self.speed_itav)        
        
        