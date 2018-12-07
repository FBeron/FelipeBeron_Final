import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('datos_observacionales.dat')
t = data[:,0]
r = data[:,1:]
r_i = r[0,:]

def model(t, r_i, param):
    n = len(t)
    r = np.zeros((n,3))
    r[0,:] = r_i
    for i in range(1,n):
        x_o = r[i-1,0]
        y_o = r[i-1,1]
        z_o = r[i-1,2]
        dt = t[i]- t[i-1]
        dx = param[0]*(y_o - x_o)
        dy = x_o*(param[1] - z_o) - y_o
        dz = x_o*y_o - param[2]*z_o
        
        r[i,0] = x_o + dx*dt
        r[i,1] = y_o + dy*dt
        r[i,2] = z_o + dz*dt
        
    return r

def logprior(param):
    superior = max(param) >= 30
    inferior = min(param) <= 0
    if superior or inferior :
        d = np.inf
    else:
        d = 0
    return d
    
def loglikelihood(x_obs, y_obs, param):
    y_i = y_obs[0,:]
    d = y_obs -  model(x_obs, y_i, param)
    d = -0.5 * np.sum(d**2) - logprior(param)
    return d

def divergence_loglikelihood(x_obs, y_obs, param):
    n_param = len(param)
    div = np.ones(n_param)
    delta = 1E-5
    for i in range(n_param):
        delta_parameter = np.zeros(n_param)
        delta_parameter[i] = delta
        div[i] = loglikelihood(x_obs, y_obs, param + delta_parameter) 
        div[i] = div[i] - loglikelihood(x_obs, y_obs, param - delta_parameter)
        div[i] = div[i]/(2.0 * delta)
    return div

def hamiltonian(x_obs, y_obs, param, p_param):
    m = 100.0
    K = 0.5 * np.sum(p_param**2)/m
    V = -loglikelihood(x_obs, y_obs, param)     
    return K + V

def leapfrog_proposal(x_obs, y_obs, param, p_param):
    N_steps = 5
    delta_t = 1E-2
    m = 100.0
    n_param = param.copy()
    n_p_param = p_param.copy()
    for i in range(N_steps):
        n_p_param = n_p_param + divergence_loglikelihood(x_obs, y_obs, param) * 0.5 * delta_t
        n_param = n_param + (n_p_param/m) * delta_t
        n_p_param = n_p_param + divergence_loglikelihood(x_obs, y_obs, param) * 0.5 * delta_t
    n_p_param = -n_p_param
    return n_param, n_p_param

def monte_carlo(x_obs, y_obs, N=5000):
    param = 30*[np.random.random(3)]
    p_param = [np.random.normal(size=3)]
    for i in range(1,N):
        
        prop_param, prop_p_param = leapfrog_proposal(x_obs, y_obs, param[i-1], p_param[i-1])
        ener_n = hamiltonian(x_obs, y_obs, prop_param, prop_p_param)
        ener_o = hamiltonian(x_obs, y_obs, param[i-1], p_param[i-1])
   
        r = min(1,np.exp(-(ener_n - ener_o)))
        alpha = np.random.random()
        if(alpha<r):
            param.append(prop_param)
        else:
            param.append(param[i-1])
        p_param.append(np.random.normal(size=3))    

    param = np.array(param)
    return param

param_chain = monte_carlo(t, r)

params = np.mean(param_chain,0)
sig_params = np.std(param_chain,0)

sigma_chain = param_chain[:,0]
rho_chain = param_chain[:,1]
beta_chain = param_chain[:,2]

tit1 = 'sigma = ' + str(params[0]) + '\n +-'  + str(sig_params[0])
tit2 = 'rho = ' + str(params[1])  + '\n +-'  + str(sig_params[1])
tit3 = 'beta = ' + str(params[2])  + '\n +-'  + str(sig_params[2])

plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.hist(sigma_chain, bins= 50)
plt.title(tit1)
plt.subplot(1,3,2)
plt.hist(rho_chain, bins= 50)
plt.title(tit2)
plt.subplot(1,3,3)
plt.title(tit3)
plt.hist(beta_chain,bins = 50)

plt.savefig('histograms.pdf')

r_pred = model(t,r_i,params)

plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.scatter(t,r[:,0])
plt.plot(t,r_pred[:,0], color ='r')
plt.title('x')
plt.xlabel('t')
plt.ylabel('x')

plt.subplot(1,3,2)
plt.scatter(t,r[:,1])
plt.plot(t,r_pred[:,1], color ='r')
plt.title('y')
plt.xlabel('t')
plt.ylabel('y')

plt.subplot(1,3,3)
plt.scatter(t,r[:,2],label = 'data')
plt.plot(t,r_pred[:,2], color ='r', label='fit')
plt.title('z')
plt.xlabel('t')
plt.ylabel('z')
plt.legend()

plt.savefig('data_fit.pdf')