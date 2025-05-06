"""
Started August 2024
by Lukas Röhrich @ Potsdam Institute for Climate Impact Research

Library for coupled differential equations to describe the complex contagion model for my Master Thesis -> Dodds-Watts meets Holmes-Newmann
"""

import numpy as np
import networkx as nx
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint      # differential equation solver

import centrality as cent
import cclib as cc

def CDE(y0,t,q,nu,m,T,p,ds,Xp,Yp):
    """
    Function to describe the coupled differential equations for the complex contagion model. Consists of three equations
    describing the dynamics of: 
    - the infected nodes                        >> ni
    - the links between two infected nodes      >> mi
    - the links between two susceptible nodes   >> ms
    Conditions:
    - 1/nu >> T
    - ds is the same for all nodes

    :param y0: Initial values for the three variables
    :type y0: list
    :param t: Time variable
    :type t: float
    :param q: rewiring probability
    :type q: float, 0 <= q <= 1
    :param nu: probability for action of a node in a given time step
    :type nu: float, 0 <= f <= 1
    :param m: number of total links divided by the number of nodes (fixed value)
    :type m: float
    :param T: memory length of the nodes
    :type T: int
    :param p: probability to receive a dose when interacting with an infected node
    :type p: float, 0 <= p <= 1
    :param ds: infection threshold (same for all nodes)
    :type ds: float
    :param Xp: list of possible values for dose probability function
    :type Xp: list
    :param Yp: list of corresponding probabilities for dose probability function
    :type Yp: list
    :return: list of the three differential
    :rtype: list
    """

    ni, mi, ms = y0     # initial values
    msi = m - mi - ms   # number of links between susceptible and infected nodes, s.c. frustrated links
    
    dnidt = (1-q) * nu * ((1-ni) * psi(m,mi,ms,T,p,ds,Xp,Yp) - ni * pis(m,mi,ms,T,p,ds,Xp,Yp))
    dmidt = (1-q) * nu * (psi(m,mi,ms,T,p,ds,Xp,Yp) * msi - pis(m,mi,ms,T,p,ds,Xp,Yp) * 2 * mi) + q * nu * ni * PIS(m,ms,mi)#msi/(2*mi + msi)
    dmsdt = (1-q) * nu * (pis(m,mi,ms,T,p,ds,Xp,Yp) * msi - psi(m,mi,ms,T,p,ds,Xp,Yp) * 2 * ms) + q * nu * (1-ni) * PSI(m,ms,mi)#msi/(2*ms + msi)
    return [dnidt, dmidt, dmsdt]

def psi(m,mi,ms,T,p,ds,Xp,Yp):
    """
    Probability for a susceptible node to become infected in a given time step.

    :param m: number of total links divided by the number of nodes (fixed value)
    :type m: float
    :param mi: number of links between two infected nodes divided by the number of nodes
    :type mi: float
    :param ms: number of links between two susceptible nodes divided by the number of nodes
    :type ms: float
    :param T: memory length of the nodes
    :type T: int
    :param p: probability to receive a dose when interacting with an infected node
    :type p: float, 0 <= p <= 1
    :param ds: infection threshold (same for all nodes)
    :type ds: float
    :param Xp: list of possible values for dose probability function
    :type Xp: list
    :param Yp: list of corresponding probabilities for dose probability function
    :type Yp: list
    :return: Probability for a susceptible node to become infected in a given time step
    :rtype: float
    """

    Psi = PSI(m,ms,mi)#((m-mi-ms)/(2*ms + m-mi-ms))  # probability for a susceptible node to choose interaction with an infected node

    s = 0
    for k in range(1,T+1):
        #X_conv,Y_conv = autoconvolve(Xp,Yp,k)
        X_conv = np.array([Xp[0]*k])  # cheap convolution of delta-distribution
        Y_conv = Yp
        s += binomial(T,k) * (p*Psi)**k * (1-p*Psi)**(T-k) * num_integrate(X_conv,Y_conv,ds,X_conv[-1])
    return s

def pis(m,mi,ms,T,p,ds,Xp,Yp):
    """
    Probability for an infected node to become susceptible in a given time step.

    :param m: number of total links divided by the number of nodes (fixed value)
    :type m: float
    :param mi: number of links between two infected nodes divided by the number of nodes
    :type mi: float
    :param ms: number of links between two susceptible nodes divided by the number of nodes
    :type ms: float
    :param T: memory length of the nodes
    :type T: int
    :param p: probability to receive a dose when interacting with an infected node
    :type p: float, 0 <= p <= 1
    :param ds: infection threshold (same for all nodes)
    :type ds: float
    :param Xp: list of possible values for dose probability function
    :type Xp: list
    :param Yp: list of corresponding probabilities for dose probability function
    :type Yp: list
    :return: Probability for an infected node to become susceptible in a given time step
    :rtype: float
    """

    Pii = PII(m,ms,mi)#((2*mi)/(2*mi + m-mi-ms)) # probability for an infected node to choose interaction with an infected node

    s = (1-p*Pii)**T                # probabilty for no received dose in the last T time-steps
    for k in range(1,T+1):
        #X_conv,Y_conv = autoconvolve(Xp,Yp,k)
        X_conv = np.array([Xp[0]*k])  # cheap convolution of delta-distribution
        Y_conv = Yp
        s += binomial(T,k) * (p*Pii)**k * (1-p*Pii)**(T-k) * num_integrate(X_conv,Y_conv,0,ds-0.01)
    return s


def PSI(m,ms,mi,err=1e-10):
    """
    WARNING: Not tested.

    Probability for a susceptible node to interact with an infected node.
    """
    if (abs(ms) <= err) & (abs(m-mi) <= err):   # no msi links
        return 0
    else:
        return (m-mi-ms)/(2*ms + m-mi-ms)
    
def PIS(m,ms,mi,err=1e-10):
    """
    WARNING: Not tested.

    Probability for an infected node to interact with a susceptible node.
    """
    if (abs(mi) <= err) & (abs(m-ms) <= err):   # no msi links
        return 0
    else:
        return (m-mi-ms)/(2*mi + m-mi-ms)
    
def PII(m,ms,mi,err=1e-10):
    """
    WARNING: Not tested.
    
    Probability for an infected node to interact with another infected node.
    """
    if (abs(mi) <= err) & (abs(m-ms) <= err):     # no msi links
        return 0
    else:
        return ((2*mi))/(2*mi + m-mi-ms)

def binomial(n, k):
    """
    WARNING: Not tested.
    
    Computes the binomail coefficient (n choose k)

    :param n: number of elements
    :type n: int
    :param k: number of elements to choose
    :type k: int
    :return: binomial coefficient (possible combinations)
    :rtype: int
    """
    if 0 <= k <= n:
        a= 1
        b=1
        for t in range(1, min(k, n - k) + 1):
            a *= n
            b *= t
            n -= 1
        return a // b
    else:
        return 0
    
def num_integrate(x,y,xs,xf):
    """
    WARNING: Not tested.
    WARNING: Not really an integration here. Working version for delta-function

    Integrates the function y(x) from xs to xf numerically

    :param x: x-values of the function
    :type x: list
    :param y: y-values of the function
    :type y: list
    :param xs: start value of the integration interval
    :type xs: float
    :param xf: end value of the integration interval
    :type xf: float
    :return: integral of the function y(x) from xs to xf
    :rtype: float
    """
    # check if x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    # cut the function to the interval [xs,xf]
    mask = (x>=xs) & (x<=xf)
    x = x[mask]
    y = y[mask]
    
    s = 0
    for i in range(len(x)):
        s += y[i]
    return s

def mim_limits(N,ni,m):
    """
    WARNING: Not tested.
    
    Computes the limits for the number of links between two infected nodes.

    :param ni: number of infected nodes divided by the number of nodes
    :type ni: float
    :param m: number of total links divided by the number of nodes (fixed value)
    :type m: float
    :param N: number of nodes
    :type N: int
    :return: lower and upper limit for the number of links between two infected nodes
    :rtype: tuple
    """
    return max(0,m + ni*(N*ni-1)/2 - 1*(N-1)/2)/m, min(m,ni*(N*ni-1)/2)/m

def msm_limits(N,ni,m,mi):
    """
    WARNING: Not tested.
    
    Computes the limits for the number of links between two susceptible nodes in dependency from mi.

    :param ni: number of infected nodes divided by the number of nodes
    :type ni: float
    :param m: number of total links divided by the number of nodes (fixed value)
    :type m: float
    :param N: number of nodes
    :type N: int
    :param mi: number of links between two infected nodes divided by the number of nodes
    :type mi: float
    :return: lower and upper limit for the number of links between two susceptible nodes
    :rtype: tuple
    """
    return max(0,m - mi - ni*(N-N*ni))/m, min(m - mi,(1-ni)*(N-N*ni-1)/2)/m



############################################################################################################
#                                        Plot Functions                                                    #
############################################################################################################

def single_run(N,beta,n,T,p,q,Xd,Yd,Xp,Yp,f,S,seed=None):
    """
    WARNING: Not tested.
    
    Runs the analytical solution for the complex contagion model for a single run.
    As inital conditions the link-type distribution of a randomly generated ER-graph is used.

    :param N: number of nodes
    :type N: int
    :param beta: density of the network, ER-graph
    :type beta: float, 0 <= beta <= 1
    :param n: fraction of initially infected nodes 
    :type n: float
    :param T: memory length of the nodes
    :type T: int
    :param p: probability to receive a dose when interacting with an infected node
    :type p: float, 0 <= p <= 1
    :param Xd: list of possible values for infection thresholds
    :type Xd: list
    :param Yd: list of corresponding probabilities for infection thresholds
    :type Yd: list
    :param Xp: list of possible values for dose probability function
    :type Xp: list
    :param Yp: list of corresponding probabilities for dose probability function
    :type Yp: list
    :param f: probability for action of a node in a given time step
    :type f: float, 0 <= f <= 1
    :param S: number of time steps
    :type S: int
    :param q: rewiring probability
    :type q: float, 0 <= q <= 1
    """

    m = beta * (N/2 - 1/2)

    # create network to get initial conditions for link-type distribution
    if seed == None:
        # - Initialize the graph, randomise the seeds
        H = nx.erdos_renyi_graph(N, beta)
        cc.InitGraph_attr(H, T, n, Xd, Yd)
        r_rank = random.sample(range(N),N)
        cent.change_seeds(H,1,r_rank)

        Mi0,Ms0,Msi0 = count_link_types(H)

        ni0 = n
        mi0 = Mi0/N
        ms0 = Ms0/N
    else:
        ni0,mi0,ms0 = seed

    y0 = [ni0, mi0, ms0]  
    t = np.linspace(0, S, S)  
    
    solutions = odeint(CDE, y0, t, args=(q, f, m, T, p, Xd[0], Xp, Yp))
    
    ni_values = solutions[:, 0]
    mi_values = solutions[:, 1]
    ms_values = solutions[:, 2]

    return ni_values,mi_values,ms_values


def hairline_data(N,beta,T,p,Xd,Yd,Xp,Yp,f,S,q,step,ni0_lim,mim0_lim=(),msm0_lim=(),verbose=False):
    """
    WARNING: Not tested.
    
    Computes the data for the hairline plot of the complex contagion model. This function takes the
    possible distribution of mi and ms links into account.
    For example: If ni is very small, there is only a very limited amount of mi links possible.
    Another example: mi+ms can not exceed m.
    
    :param N: number of nodes
    :type N: int
    :param beta: density of the network, ER-graph
    :type beta: float, 0 <= beta <= 1
    :param T: memory length of the nodes
    :type T: int
    :param p: probability to receive a dose when interacting with an infected node
    :type p: float, 0 <= p <= 1
    :param Xd: list of possible values for infection thresholds
    :type Xd: list
    :param Yd: list of corresponding probabilities for infection thresholds
    :type Yd: list
    :param Xp: list of possible values for dose probability function
    :type Xp: list
    :param Yp: list of corresponding probabilities for dose probability function
    :type Yp: list
    :param f: probability for action of a node in a given time step
    :type f: float, 0 <= f <= 1
    :param S: number of time steps
    :type S: int
    :param q: rewiring probability
    :type q: float, 0 <= q <= 1
    :param ni_step: step size for the number of infected nodes
    :type ni_step: float
    :param step: step sizes for the number of infected nodes, the number of links between two infected nodes and the number of links between two susceptible nodes
    :type step: tuple
    :param ni0: range for the number of infected nodes
    :type ni0: tuple
    :param mim0: range for the number of links between two infected nodes, if empty the function will compute the limits
    :type mim0: tuple
    :param msm0: range for the number of links between two susceptible nodes, if empty the function will compute the limits
    :type msm0: tuple
    :param verbose: print additional information
    :type verbose: bool
    :return: data for the hairline plot


    """

    m = beta * (N/2 - 1/2)
    if verbose: print(f'm: {m}')

    # get step sizes
    ni_step = step[0]
    mim_step = step[1]
    msm_step = step[2]

    ni0_start,ni0_end = ni0_lim
    if ni0_end == ni0_start:
        ni0_series = np.array([ni0_start])
    else:
        ni0_series = np.arange(ni0_start,ni0_end,ni_step,dtype=float)   # end point NOT included
        ni0_series = np.append(ni0_series,ni0_end-ni0_start)            # append value close to end point

    # estimate the array sizes
    ni0_R = len(ni0_series)
    mim0_R = int(1/mim_step)+1
    msm0_R = int(1/msm_step)+1

    # make space for the data
    ni_values = np.full((ni0_R,mim0_R,msm0_R,S),np.nan)
    mi_values = np.full((ni0_R,mim0_R,msm0_R,S),np.nan)
    ms_values = np.full((ni0_R,mim0_R,msm0_R,S),np.nan)

    if verbose: print(f'\nni0_series: {np.round(ni0_series,5)}')
    for i in range(len(ni0_series)):#ni0_R):

        ni0 = round(ni0_series[i],5)
        # - define mim0 range
        if len(mim0_lim) == 0:
            mim0_start,mim0_end = mim_limits(N,ni0,m)
        else:
            mim0_start,mim0_end = mim0_lim
        mim0_series = np.arange(mim0_start,mim0_end,mim_step,dtype=float)
        mim0_series = np.append(mim0_series,mim0_end-mim0_start)
        if verbose: print(f'{i+1} of {len(ni0_series)}\t mi0_series: {[round(x*m,5) for x in mim0_series]}')
        for j in range(len(mim0_series)):

            mim0 = round(mim0_series[j],5)
            # - define mim0 range
            if len(msm0_lim) == 0:
                msm0_start,msm0_end = msm_limits(N,ni0,m,mim0*m)
            else:
                msm0_start,msm0_end = msm0_lim
            msm0_series = np.arange(msm0_start,msm0_end,msm_step,dtype=float)
            msm0_series = np.append(msm0_series,msm0_end-msm0_start)
            if verbose: print(f'\t{j+1} of {len(mim0_series)}\t ms0_series: {[round(x*m,5) for x in msm0_series]}')
            for k in range(len(msm0_series)):

                msm0 = round(msm0_series[k],5)

                mi0 = round(mim0*m,5)
                ms0 = round(msm0*m,5)

                y0 = [ni0,mi0,ms0]
                #if verbose: print(f'\t\t\t ni0: {ni0}, mi0: {mi0}, ms0: {ms0}')
                t = np.linspace(0, S, S)  
                
                solutions = odeint(CDE, y0, t, args=(q, f, m, T, p, Xd[0], Xp, Yp))
                
                ni_values[i,j,k] = solutions[:, 0]
                mi_values[i,j,k] = solutions[:, 1]
                ms_values[i,j,k] = solutions[:, 2]

    return ni_values,mi_values,ms_values
                
def triple_plot2D(ni_values,mi_values,ms_values,N,beta,T,p,Xd,f,q,save_path=None):
    """
    WARNING: Not tested.
    
    """

    ni0_R,mim0_R,msm0_R,S = ni_values.shape
    m = beta * (N/2 - 1/2)

    # make 3 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    cmap = mpl.colormaps.get_cmap('rainbow')  # viridis is the default colormap for imshow
     
    axs[0,0].set_title(r'$m_I$ and $m_S$')
    axs[0,1].set_title(r'$n_I$ and $m_I$')
    axs[1,0].set_title(r'$n_I$ and $m_S$')

    # - plot mI and mS
    for i in range(ni0_R):
        for j in range(mim0_R):
            for k in range(msm0_R):
                try:
                    axs[0,0].plot(mi_values[i,j,k]/m, ms_values[i,j,k]/m,color=cmap(ni_values[i,j,k,0]), linewidth=0.5,alpha=0.5)
                    axs[0,0].plot(mi_values[i,j,k,-1]/m,ms_values[i,j,k,-1]/m,marker='x',color=cmap(ni_values[i,j,k,0]),alpha=0.5)
                except:
                    pass
    axs[0,0].set_xlabel(r'$\frac{m_I}{m}$',fontsize=16)
    axs[0,0].set_ylabel(r'$\frac{m_S}{m}$',fontsize=16)
    axs[0,0].grid()

    # Add colorbar to the first subplot
    norm_ni = mpl.colors.Normalize(vmin=0, vmax=1)
    sm_ni = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_ni)
    sm_ni.set_array([])
    cbar_0 = fig.colorbar(sm_ni, ax=axs[0,0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar_0.set_label(r'$n_I$')

    # - plot nI and mI
    for i in range(ni0_R):
        for j in range(mim0_R):
            for k in range(msm0_R):
                try:
                    axs[0,1].plot(ni_values[i,j,k], mi_values[i,j,k]/m,color=cmap(ms_values[i,j,k,0]/m), linewidth=0.5,alpha=0.5)
                    axs[0,1].plot(ni_values[i,j,k,-1],mi_values[i,j,k,-1]/m,marker='x',color=cmap(ms_values[i,j,k,0]/m),alpha=0.5)
                except:
                    pass
    axs[0,1].set_xlabel(r'$n_I$',fontsize=16)
    axs[0,1].set_ylabel(r'$\frac{m_I}{m}$',fontsize=16)
    axs[0,1].grid()

    # Add colorbar to the second subplot
    norm_ms = mpl.colors.Normalize(vmin=0, vmax=1)
    sm_ms = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_ms)
    sm_ms.set_array([])
    cbar_1 = fig.colorbar(sm_ms, ax=axs[0,1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar_1.set_label(r'$\frac{m_s}{m}$')

    # - plot nI and mS
    for i in range(ni0_R):
        for j in range(mim0_R):
            for k in range(msm0_R):
                try:
                    axs[1,0].plot(ni_values[i,j,k], ms_values[i,j,k]/m,color=cmap(mi_values[i,j,k,0]/m), linewidth=0.5,alpha=0.5)
                    axs[1,0].plot(ni_values[i,j,k,-1],ms_values[i,j,k,-1]/m,marker='x',color=cmap(mi_values[i,j,k,0]/m),alpha=0.5)
                except:
                    pass
    axs[1,0].set_xlabel(r'$n_I$',fontsize=16)
    axs[1,0].set_ylabel(r'$\frac{m_S}{m}$',fontsize=16)
    axs[1,0].grid()

    # Add colorbar to the third subplot
    norm_mi = mpl.colors.Normalize(vmin=0, vmax=1)
    sm_mi = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_mi)
    sm_mi.set_array([])
    cbar_2 = fig.colorbar(sm_mi, ax=axs[1,0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar_2.set_label(r'$\frac{m_I}{m}$')


    # Add text block to the fourth subplot
    axs[1, 1].axis('off')  # Turn off the axis
    text = f'N={N}\n'+r'$\beta$'+f'={beta}\nm={m}\nf={f}\np={p}\nq={q}\nT={T}\nds={Xd[0]}\nS={S}'
    axs[1, 1].text(0.5, 0.5, text, ha='center', va='center', fontsize=16, transform=axs[1, 1].transAxes)

    # Adjust the spacing between subplots
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()

def single_plot2D(X_values,Y_values,Z_values,X_label=None,Y_label=None,Z_label=None,legend=None,save_path=None):
    """
    WARNING: Not tested.
    
    Creates as interactvie 2D plot.

    :param X_values: x-values
    :type X_values: np.array
    :param Y_values: y-values
    :type Y_values: np.array
    :param Z_values: z-values
    :type Z_values: np.array
    :param X_label: label for the x-axis
    :type X_label: str
    :param Y_label: label for the y-axis
    :type Y_label: str
    :param Z_label: legend for the z-values
    :type Z_label: str
    :param legend: additional information to pu into the legend
    :type legend: str
    :param save_path: path to save the plot
    :type save_path: str
    """

    #plt.ion() # turn on interactive mode

    ni0_R,mim0_R,msm0_R,S = X_values.shape
    #m = beta * (N/2 - 1/2)

    fig, ax = plt.subplots(figsize=(15,10))
    cmap = mpl.colormaps.get_cmap('rainbow')  # viridis is the default colormap for imshow

    plt.plot([],[],linewidth=0,label=legend)

    # - plot X and Y
    for i in range(ni0_R):
        for j in range(mim0_R):
            for k in range(msm0_R):
                try:
                    plt.plot(X_values[i,j,k], Y_values[i,j,k],color=cmap(Z_values[i,j,k,0]), linewidth=0.5,alpha=0.5)
                    plt.plot(X_values[i,j,k,-1],Y_values[i,j,k,-1],marker='x',color=cmap(Z_values[i,j,k,0]),alpha=0.5)
                except:
                    pass
    plt.xlabel(X_label,fontsize=16)
    plt.ylabel(Y_label,fontsize=16)
    plt.grid()
    if legend != None:
        plt.legend(fontsize=16)

    # Add colorbar to the first subplot
    if Z_label != None:
        norm_ni = mpl.colors.Normalize(vmin=0, vmax=1)
        sm_ni = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_ni)
        sm_ni.set_array([])
        cbar = fig.colorbar(sm_ni, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(Z_label,fontsize=16)

    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()



############################################################################################################
#                                       Helper Functions                                                   #
############################################################################################################

def deltadist(x,d=1):
    """
    WARNING: Not tested.
    
    Takes a list of x-values and returns a list of y-values with a delta-distribution at d.

    :param x: x-values
    :type x: list
    :param d: position of the delta-distribution
    :type d: float
    :return: y-values
    :rtype: list
    """

    y = np.zeros(len(x))            # initialize y-list with zeros
    y[np.argmin(np.abs(x-d))] = 1   # find index i of x-list closest to d and set y[i] to 1
    return y

def count_link_types(H):
    """
    WARNING: Not tested.
    
    Checks all the edges of graph H and counts the number of different types of links.
    Types:
    - Mi: links between two infected nodes
    - Ms: links between two susceptible nodes
    - Msi: links between a susceptible and an infected node, s.c. frustrated links

    :param H: Network with status attributes
    :type H: networkx.Graph
    :return: Number of links of each type
    :rtype: tuple
    """

    Mi0 = 0                 # infected links    
    Ms0 = 0                 # susceptible links
    Msi0 = 0                # frustrated links
    for e in H.edges():     # iterate through all edges
        if H.nodes[e[0]]['status'] == 1:
            if H.nodes[e[1]]['status'] == 1:
                Mi0 += 1    # infected link
            else:
                Msi0 += 1   # frustrated link
        else:
            if H.nodes[e[1]]['status'] == 1:
                Msi0 += 1   # frustrated link
            else:
                Ms0 += 1    # susceptible link
    return Mi0,Ms0,Msi0