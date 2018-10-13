# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:05:15 2018

@author: Emanuele

Simple example on how to compute the derivative check
"""

import numpy as np

def exp(x, w, b):
    return np.exp(np.dot(x.T,w).T + b);

def d_exp(x, w, b):
    return np.tile(exp(x, w, b), x.shape[0]).T*x, exp(x, w, b);

def check_grad(dw, db, x, w, b):
    
    eps = 1e-5;
    grad_by_def = np.array([]);
    grad_by_formula = np.concatenate((dw.flatten(), db.flatten()));
    
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i][j] += eps;
            g_plus = exp(x, w, b)[j];
            
            w[i][j] -= 2*eps;
            g_minus = exp(x, w, b)[j];
            w[i][j] += eps;
            
            grad_by_def = np.append(grad_by_def, (g_plus-g_minus)/(2*eps));
            
    for i in range(b.shape[0]):
        b[i] += eps;
        g_plus = exp(x, w, b)[i];
        
        b[i] -= 2*eps;
        g_minus = exp(x, w, b)[i];
        b[i] += eps;
        
        grad_by_def = np.append(grad_by_def, (g_plus-g_minus)/(2*eps));
              
    error = np.linalg.norm(grad_by_def-grad_by_formula)/np.linalg.norm(grad_by_formula);
    
    return error, grad_by_def, grad_by_formula;

if __name__ == "__main__":
    
    w = np.random.rand(5,10);
    b = np.random.rand(10,1);    
    x = np.random.rand(5,1);
    
    print(check_grad(*(d_exp(x, w, b)), x, w, b));            
            