import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm
from random import randint
from tabulate import tabulate
import seaborn as sns
from scipy import stats
import scipy.optimize as optimize
import pandas as pd


seed = 1
# seed = randint(0, 100000)                                                         # Create random seed
np.random.seed(seed)                                                                # Set seed

# T         = End  time
# NoOfSteps = Number of steps in the time grid
# NoOfRates = Number of rates
# insCorr   = Matrix containing instantaneous correlation
def GenerateBM(T, NoOfSteps, NoOfRates, insCorr):
    """"Generate Brownian motions."""
    dt = T / NoOfSteps                                                              # Discretization grid
    
    Z = np.random.normal(0, 1, [NoOfRates, NoOfSteps])                              # Create standard normal variables
    Zanti = -Z                                                                      # Antithetic variables
    W = np.zeros([NoOfRates, NoOfSteps+1])                                          # Initialize Brownian motion
    Wanti = np.zeros([NoOfRates, NoOfSteps+1])                                      # Initialize Brownian motion

    C = insCorr                                                                     # Obtain correlation structure for the Brownian motions
    L = np.linalg.cholesky(C)                                                       # Apply Cholesky decomposition

    for i in range(NoOfSteps):
        # Calculate the BM for every forward rate per time step i       
        W[:, i+1] = W[:, i] + (np.power(dt, 0.5) * Z[:, i])
        Wanti[:, i+1] = Wanti[:, i] + (np.power(dt, 0.5) * Zanti[:, i])

    W = L @ W                                                                       # Correlate the BM's
    Wanti = L @ Wanti                                                               # Correlate the BM's

    return W, Wanti                                                                 # Return the generated Brownian motion

# NoOfSteps = Number of steps in the time grid
# NoOfRates = Number of rates
# IV        = Given instantaneous volatilities from the market
def insVol(NoOfSteps, NoOfRates, IV): 
    """Generate the instantaneous volatility matrix."""
    V = np.zeros([NoOfRates, NoOfSteps]) 
       
    tenor_steps = NoOfSteps / NoOfRates                                             # Steps in between tenor points
    
    for i in range(NoOfSteps):                                                      # Loop per time step
        for j in range(NoOfRates):                                                  # Loop per forward rate
            if i == 0:
                V[j,:] = IV[j]                                                      # Set every time step equal to the initialized IV
            if i >= tenor_steps*j:
                V[0:j,i:] = np.nan                                                  # If the forward rate is 'dead' reset the value to NaN

    return V                                                                        # Return the instantaneous volatility matrix

# NoOfRates = Number of rates
def insCorr(NoOfRates):
    """Generate the instantaneous correlation matrix."""    
    N = NoOfRates                                                                   # N is the total number of rates
    C = np.ones((N, N))                                                             # Initialize the correlation matrix
    
    # Self-picked the value of 0.95
    np.fill_diagonal(C[:,1:], 0.95*np.ones(N-1))                                    # Set the value of the diagonal right from the true diagonal equal to 0.95  
    
    for i in range(N):
        for j in range(N):
            if j - i == 1:                                                          # Make the matrix symmetric
                C[j,i] = C[i,j]
            if j - i > 1:                                                           # Fill in the rest of the matrix
                k = i
                while k < j:
                    C[i,j] = C[i,j] * C[k,k+1]
                    C[j,i] = C[i,j]
                    k = k + 1   
        
    return C                                                                        # Return the correlation matrix

# NoOfRates      = Number of rates
# Initial_market = Initial market rates
def inFor(NoOfRates, initial_market): 
    """Generate the initial forward rates."""      
    L = initial_market[:NoOfRates].to_numpy()
    return L                                                                        # Return the forward rates

# NoOfSteps     = Number of time steps
# NoOfRat       = Number of rates
# T             = End time
# tau           = Difference between time steps
# V             = Instantaneous volatility matrix
# L0            = Initial forward rates
# theta         = Shift values
def Generate_BWR_Log_Eul(NoOfSteps, NoOfRat, T, tau, V, L0, theta):
    # start = timeit.default_timer()   
    """Generate backward rates using the log-Euler method."""
    """You can actually see this as generating forward rates then in the next function we put the right values at the correct places"""
    # Obtain the initial values needed for the simulation
    IC = insCorr(NoOfRat)                                                            # Instantaneous correlation matrix
    BM, BManti = GenerateBM(T, NoOfSteps, NoOfRat, IC)                               # Generated Brownian motions

    L0_original = L0                                                                 # Initial negative rates
    L0_shifted  = L0 + theta                                                         # Shifted positive rates
        
    BWR_org = np.zeros([NoOfRat, NoOfSteps + 1])                                     # Create matrices for the backward/forward rates
    BWR_org[:,0] = np.transpose(L0_original)                                         # Set initial values
    BWR_shift = np.zeros([NoOfRat, NoOfSteps + 1])                                   # Create matrices for the backward/forward rates
    BWR_shift[:,0] = np.transpose(L0_shifted)                                        # Set initial values
        
    BWR_org_anti = np.zeros([NoOfRat, NoOfSteps + 1])                                # Do the same for the antithetic variables
    BWR_org_anti[:,0] = np.transpose(L0_original) 
    BWR_shift_anti = np.zeros([NoOfRat, NoOfSteps + 1]) 
    BWR_shift_anti[:,0] = np.transpose(L0_shifted)  

    # Calculate the constant over time C matrix. Only constant since IV constant
    fa = np.outer(V[:,0] , V[:,0]) * IC[:, :]
    fb = np.outer(V[:,0] , V[:,0]) * IC[:, :]
    Cii = np.diagonal(tau * ((fa + fb) / 2)) 
    C = tau * ((fa + fb) / 2) 
    for time in range(NoOfSteps):
        '''Loop per time step'''
        Z = V[:, 0] * (BM[: , time+1] - BM[: , time])                          # Note: I just use the first volatility value since it is constant over time
        Zanti = V[:, 0] * (BManti[: , time+1] - BManti[: , time])              # Note: I just use the first volatility value since it is constant over time

        drift_approx       = (tau * (BWR_org[:,time] + theta)) / (1 + tau * BWR_org[:,time]) * C            # Start drift approximation
        drift_approx_anti  = (tau * (BWR_org_anti[:,time] + theta)) / (1 + tau * BWR_org_anti[:,time]) * C  # Start drift approximation
      
        matrix1 = np.tril(drift_approx[:,time:],-time)                         # Help matrix
        matrix2 = np.tril(drift_approx_anti[:,time:],-time)                    # Help matrix
        
        X      = np.nansum(matrix1, axis=1)                                    # Drift approximation
        X_anti = np.nansum(matrix2, axis=1)                                    # Drift approximation
            
        
        BWR_shift[:,time+1]      = BWR_shift[:,time] * np.exp(X + (-0.5 * Cii) + Z)                       # Calculate the next step of the shifted lognormal rates
        BWR_org[:,time+1]        = BWR_shift[:,time+1] - theta                                            # Shift back to obtain the original rates, needed for the dynamics of the shifted rates
        BWR_shift_anti[:,time+1] = BWR_shift_anti[:,time] * np.exp(X_anti + (-0.5 * Cii) + Zanti)         # Calculate the next step of the shifted lognormal rates
        BWR_org_anti[:,time+1]   = BWR_shift_anti[:,time+1] - theta                                       # Shift back to obtain the original rates, needed for the dynamics of the shifted rates   
               
        # Predictor-Corrector Method
        approxFRW          = 0.5 * ((BWR_org[:,time+1] + theta) + (BWR_org[:,time] + theta))
        approxFRW_org      = 0.5 * ((BWR_org[:,time+1]) + (BWR_org[:,time]))  
        approxFRW_anti     = 0.5 * ((BWR_org_anti[:,time+1] + theta) + (BWR_org_anti[:,time] + theta))
        approxFRW_org_anti = 0.5 * ((BWR_org_anti[:,time+1]) + (BWR_org_anti[:,time]))
    
        drift_approx      = (tau * approxFRW) / (1 + tau * approxFRW_org) * C
        drift_approx_anti = (tau * approxFRW_anti) / (1 + tau * approxFRW_org_anti) * C
        
        matrix1 = np.tril(drift_approx[:,time:],-time)
        matrix2 = np.tril(drift_approx_anti[:,time:],-time)       
        
        X      = np.nansum(matrix1, axis=1)
        X_anti = np.nansum(matrix2, axis=1)

    
        BWR_shift[:,time+1]      = BWR_shift[:,time] * np.exp(X + (-0.5 * Cii) + Z)                # Calculate the next step of the shifted lognormal rates
        BWR_org[:,time+1]        = BWR_shift[:,time+1] - theta                                     # Shift back to obtain the original rates, needed for the dynamics of the shifted rates
        BWR_shift_anti[:,time+1] = BWR_shift_anti[:,time] * np.exp(X_anti + (-0.5 * Cii) + Zanti)  # Calculate the next step of the shifted lognormal rates
        BWR_org_anti[:,time+1]   = BWR_shift_anti[:,time+1] - theta                                # Shift back to obtain the original rates, needed for the dynamics of the shifted rates
            
    return BWR_org, BWR_org_anti, BWR_shift, BWR_shift_anti, IC                                                                   # Return backward rates

# NoOfSteps     = Number of time steps
# NoOfRates     = Number of rates
# T             = End time
# tau           = Difference between time steps
# V             = Instantaneous volatility matrix
# L             = Initial forward rates
# eul_steps     = Number of euler discretization steps
# cap_shift     = Shifted to obtain positive rates
def Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, V, L, eul_steps, cap_shift):
    """Generate backward-looking forward rates"""
    """Important to remember!! We have a different starting date / today date. Difference is exactly tau. Hence first backward rate is over [T_1,T_2]"""
    BWR_original, BWRanti_original, BWR_shifted, BWRanti_shifted, IC  = Generate_BWR_Log_Eul(NoOfSteps, NoOfRates, T, tau, V, L, cap_shift)                   # Obtain backward rates which were generated using the log_euler method

    # Initialize the backward rates
    BW_rate_original      = np.zeros([NoOfRates, NoOfSteps + eul_steps + 1])                                 
    BW_rate_anti_original = np.zeros([NoOfRates, NoOfSteps + eul_steps + 1]) 

    BW_rate_shifted       = np.zeros([NoOfRates, NoOfSteps + eul_steps + 1])                                 
    BW_rate_anti_shifted  = np.zeros([NoOfRates, NoOfSteps + eul_steps + 1]) 

    # Every rate needs it's own 'x-axis'
    x_axis = np.zeros([NoOfRates, NoOfSteps + eul_steps + 1])
       
    # Set values up to time T_(j-1) to the same value as forward rates
    """For t <= T_j the backward rates are equal to the log-Euler rates this
    is the same as the generated forward rates for the LMM. So this creates the forward rates"""
    for i in range(NoOfRates):
        BW_rate_original[i, 0:i+2]      = BWR_original[i, 0:i+2]
        BW_rate_anti_original[i, 0:i+2] = BWRanti_original[i, 0:i+2]
        
        BW_rate_shifted[i, 0:i+2]      = BWR_shifted[i, 0:i+2]
        BW_rate_anti_shifted[i, 0:i+2] = BWRanti_shifted[i, 0:i+2]
        x_axis[i, 0:i+2]               = np.arange(0, i+2) * tau

    # Euler discretization size
    dt = tau / eul_steps                                                                  # time step
        
    BM_new, BManti_new = GenerateBM(tau, eul_steps, NoOfRates, IC)                      # Create new Brownian motions    

    A_org = A = np.arange(NoOfRates).reshape((NoOfRates,1)) + 1                           # A is a filter which can be put over matrices obtaining the correct rows and columns we need to change
    for steps in range(eul_steps):                                                        # Loop per Euler step    
        x_axis[np.arange(A.shape[0])[:,None], (A+1)] = (x_axis[np.arange(A.shape[0])[:,None], (A)] + dt)       # Next value for the x-axis
        Tj = x_axis[np.arange(A_org.shape[0])[:,None], (A_org)] + tau                                          # Tj per rate
        t = x_axis[np.arange(A.shape[0])[:,None], (A)]                                                         # t per rate, Note!! This does differ per rate due to mixed combination of methods

        # Apply gamma function
        z1 = Tj - t
        z1[z1 < 0] = 0
        gamma_rate = z1 / (Tj - (Tj-tau))
        gamma_rate[gamma_rate > 1] = 1

        X = Xanti = 0

        if steps == 0: # No need for for loop since only 1 time
            # No need for the gamma function since it will be 1! Only the case for our chosen gamma function
            # No need for IC since we have just the same of j till j so i always equal to 1.            
            Y = (tau * V[:,0] * (BW_rate_original[np.arange(A.shape[0])[:,None], (A)] + cap_shift) * gamma_rate)
            Z = 1 + tau * BW_rate_original[np.arange(A.shape[0])[:,None], (A)]
            X += Y / Z

            Yanti = np.diagonal(tau * V[:,0] * (BW_rate_anti_original[np.arange(A.shape[0])[:,None], (A)] + cap_shift) * gamma_rate)
            Zanti = 1 + tau * BW_rate_anti_original[np.arange(A.shape[0])[:,None], (A)]
            Xanti += Yanti / Zanti  
            
            # Values that are present in the summation of the formula
            X = np.diagonal(X)
            Xanti = np.diagonal(Xanti)
            
        BW_rate_shifted[np.arange((A+1).shape[0])[:,None], (A+1)] = np.reshape(np.diagonal(BW_rate_shifted[np.arange(A.shape[0])[:,None], (A)] + V[:,0] * gamma_rate * BW_rate_shifted[np.arange(A.shape[0])[:,None], (A)] * X * dt + \
            V[:,0] * BW_rate_shifted[np.arange(A.shape[0])[:,None], (A)] * gamma_rate * (BM_new[:,steps+1] - BM_new[:,steps])), (NoOfRates,1))                                      # Perform the Euler discretization
        BW_rate_original[np.arange((A+1).shape[0])[:,None], (A+1)] = BW_rate_shifted[np.arange(A.shape[0])[:,None], (A+1)] - cap_shift                  # Shift back the rates

        
        BW_rate_anti_shifted[np.arange((A+1).shape[0])[:,None], (A+1)] = np.reshape(np.diagonal(BW_rate_anti_shifted[np.arange(A.shape[0])[:,None], (A)] + V[:,0] * gamma_rate * BW_rate_anti_shifted[np.arange(A.shape[0])[:,None], (A)] * Xanti * dt + \
            V[:,0] * BW_rate_anti_shifted[np.arange(A.shape[0])[:,None], (A)] * gamma_rate * (BManti_new[:,steps+1] - BManti_new[:,steps])), (NoOfRates,1))                         # Perform the Euler discretization
        BW_rate_anti_original[np.arange((A+1).shape[0])[:,None], (A+1)] = BW_rate_anti_shifted[np.arange(A.shape[0])[:,None], (A+1)] - cap_shift       # Shift back the rates
    
        A = A+1

    # Define values after the accrual period
    BW_rate_original[BW_rate_original == 0] = BW_rate_anti_original[BW_rate_anti_original == 0] = np.nan
    BW_rate_shifted[BW_rate_shifted == 0]   = BW_rate_anti_shifted[BW_rate_anti_shifted == 0]   = np.nan

    return BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis 

# Notional   = Notional of the contract
# tau        = Difference between tenor points
# T          = Reset date of the caplet
# K          = Strike price of the caplet
# V          = Instantaneous volatility matrix
# L          = Initional forward rates
# cap_shift  = Shift given by the market
# DF         = Discount factor for period [T0, T1]
"""This is because for the log-Euler method the backward and forward rates are the same
hence we do not work with the spotrate yet which is equal to the start of the first backward rates
I just define a seperate shift for that. There are also other solutions.""" 
def cap_price_Black(Notional, tau, T, K, V, L, cap_shift, DF):  
    """Calculate the price of a caplet using Black's formula"""
    
    num_of_caps = int(T/tau)                                                                    # Number of caplets we want to validate
    cap_price = np.zeros(num_of_caps)                                                           # Save caplet prices
    
    for resetT in range(1, num_of_caps + 1):                                                    # Apply Black's formula for every caplet                   
        helpval = (tau ** 3) / (tau ** 2)                                                       # We want the caplet price at time zero hence no maximum needed
    
        R = L[resetT-1] + cap_shift
        K_hat = K + cap_shift
        vsqr = (resetT * tau + 1 / 3 * helpval) * V[resetT-1, 0] * V[resetT-1, 0] 
            
               
        d1 = (np.log(R / K_hat) + (vsqr / 2)) / (np.sqrt(vsqr)) 
        d2 = d1 - (np.sqrt(vsqr))
        
        P = DF[0]
        for i in range(0,resetT): 
            # Loop to calculate the value of P(0,T_n)
            P = P * (1 / (1 + tau * L[i]) )
         
         
        cap_price[resetT-1] = Notional * P * tau * (R * norm.cdf(d1) - K_hat * norm.cdf(d2) )
        if cap_price[resetT-1] / Notional < 10 ** -8:                                             # I take this as such a small value that we assume that the capprice is equal to zero
            cap_price[resetT-1] = 0


    return cap_price

# N          = Notional
# tau        = Difference between tenor points
# resetT     = Reset date of the caplet
# K          = Strike price of the caplet
# V          = Instantaneous volatility matrix of the large time step method
# L          = Initial forward rates
# NoOfSteps  = Number of steps used
# NoOfRates  = Number of rates that will be simulated
# T          = End date
# M          = Number of Monte Carlo simulations
# eul_steps  = Number of Euler discretization steps
# cap_shift  = SHift given by the market
# DF         = Discount factor for first period
def cap_price_MC(N, tau, K, V, L, NoOfSteps, NoOfRates, T, M, eul_steps, cap_shift, DF):
    """Calculate the price of a caplet using Monte Carlo simulation"""   
    cappriceBack = np.zeros([NoOfRates,M])
    A = np.arange(NoOfRates).reshape((NoOfRates,1)) + eul_steps + 1               # Filter to get the correct rows and columns
    A_disc = np.arange(NoOfRates).reshape((NoOfRates,1)) + 1                      # Filter for obtaining discount values
    
    for i in range(M):
        [BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis] = Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, V, L, eul_steps, cap_shift)                # Obtain backward rates using a large time step
        # Calculate payoffs and set negative values to zero (=max operator)
        payoffLS                         = (BW_rate_original[np.arange(A.shape[0])[:,None], (A)] - K) * N * tau
        payoffLS[payoffLS < 0]           = 0
        payoffLS_anti                    = (BW_rate_anti_original[np.arange(A.shape[0])[:,None], (A)] - K) * N * tau
        payoffLS_anti[payoffLS_anti < 0] = 0

        # Discount factors
        rates         = (1 + tau * BW_rate_original[np.arange(A_disc.shape[0])[:,None], (A_disc)])
        discount      = np.zeros(len(rates))
        rates_anti    = (1 + tau * BW_rate_anti_original[np.arange(A_disc.shape[0])[:,None], (A_disc)])
        discount_anti = np.zeros(len(rates_anti))

        for j in range(len(rates)):
            discount[j]      = np.prod(rates[:j+1,0]) * (1 / DF[0])                                      # Need to multiply with (1 / DF) since this is the discount factor for the first period but need to revert it back
            discount_anti[j] = np.prod(rates_anti[:j+1,0]) * (1 / DF[0]) 
            
        discount      = np.reshape(discount, (NoOfRates,1))
        discount_anti = np.reshape(discount_anti, (NoOfRates,1))
                      
        cappriceBack[:,i] = (np.reshape(np.divide(payoffLS , discount),(NoOfRates,)) + np.reshape(np.divide(payoffLS_anti , discount_anti),(NoOfRates,))) * 0.5
        
    # Calculate standard errors
    se_Back  = np.sqrt(np.var(cappriceBack, axis=1, ddof=1)) / np.sqrt(M)       
    
    return np.sum(cappriceBack, axis=1) / M, se_Back

# NoOfSteps  = Number of steps used
# NoOfRates  = Number of rates that will be simulated
# T          = End date
# tau        = Difference between tenor points
# V          = Instantaneous volatility matrix of the large time step method
# L          = Initial forward rates
# eul_steps  = Number of Euler discretization steps
# cap_shift  = Shift given by the market
def density_plot(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, cap_shift):
    """This function makes a 3D plot of different paths for a rate and also density plots at different time intervals"""
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    NoOfLines = 500                                                             # Number of different paths you want to plot
    
    # One time calculation to obtain some sizes
    BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, cap_shift)
    
    # Define the y-axis and z-axis for the plotting the rates
    zline = np.linspace(0, 0, len(x_axis[0,:]))
    yline = x_axis[NoOfRat-1,:]
    
    # Define limits for the x-axis and z-axis
    left_lim  = -0.015
    right_lim = 0.005
    ax.set_xlim(left_lim, right_lim)  
    ax.set_zlim(0, 500)
    
    data = np.zeros([T,NoOfLines])                                              # Save data to make density plots
       
    for i in range(NoOfLines):
        # Plot different paths
        BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, cap_shift)
        xline = BW_rate_original[NoOfRat-1,:]
        
        if np.size(xline[xline>right_lim]) == 0 and i < 50: # Make sure lines don't cross the x limits
            ax.plot3D(xline, yline, zline, 'blue')
        
        # Obtain values for density plots
        for j in range(T):
            col = int((1/tau) * (j+1))
            if j == (T-1):
                col = -1
            data[j,i] = BW_rate_original[NoOfRat-1, col]
    
    for j in range(T):
        # Make density plots
        col = int((1/tau) * (j+1))
        if j == (T-1):
            col = -1
        [xdata, ydata] = get_hist_lines(data[j,:])
           
        # Distinguish positive and negative values
        idx_pos_1 = np.where((xdata >= 0) & (xdata < right_lim))  ; pos_x = xdata[idx_pos_1] ; pos_y = ydata[idx_pos_1] 
        idx_neg_1 = np.where((xdata <= 0) & (xdata > left_lim))   ; neg_x = xdata[idx_neg_1] ; neg_y = ydata[idx_neg_1]
        
        xpos = x_axis[NoOfRat-1, col]
                             
        ax.plot3D(pos_x,np.linspace(xpos,xpos,len(pos_x)),pos_y, 'red')    
        ax.plot3D(neg_x,np.linspace(xpos,xpos,len(neg_x)),neg_y, color='black')
    
    ax.set_xlabel('rate')
    ax.set_ylabel('Time (years)')
    ax.set_zlabel('density')
    ax.set_title('R(t) paths and density plots in a negative rate environment')
    
    ax.view_init(35, -35)    
    return

# data = Histogram data
def get_hist_lines(data):
    """This function gahters the correct data for the denisty plots"""
    plt.figure()
    xdata = sns.distplot(data).get_lines()[0].get_data()[0]
    ydata = sns.distplot(data).get_lines()[0].get_data()[1]
    plt.close()
    
    return xdata, ydata
    

def mainCalculation():
    """Run the program to generate backward rates."""    
    tau         = 0.25                                                              # Difference between tenor points
    T           = 1                                                                 # Time horizon
    eul_steps   = 50                                                                # Euler discretization steps between two tenor points    
    
    NoOfSteps   = int(T / tau)                                                      # Number of time-steps                                                    
    NoOfRat     = int(T / tau)                                                      # Number of backward rates we want to generate

    # Obtain the correct data (Initial rates, Instantaneous volatilities)
    df = pd.read_excel (r'C:\Users\hackt\Documents\Thesis Rabobank\Python\Caplets_3Months.xlsx')

    help1 = df.iloc[17:,:]
    data = help1.iloc[:,[1,2,5,7]]
    data.reset_index(drop=True, inplace=True)
    data.columns = ['Date', 'CapletForward', 'DF', 'IV']

    V         = insVol(NoOfSteps, NoOfRat, data.IV) 
    L         = inFor(NoOfRat, data.CapletForward)  / 100                                        # Initial rates
    cap_shift = 3 / 100
    # Obtain the forward-looking backward rates and their corresponding x-axis
    BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, cap_shift)

       
    "Plotting the rates"
    # Obtain labels and locations for the x-axis
    labels = np.zeros(NoOfRat+2) ; locs = np.zeros(NoOfRat+2) 
    hv = 0
    for i in range(NoOfRat+2):
        labels[i] = hv
        locs[i] = i * tau
        hv += 0.25

    fig, ax = plt.subplots()
    for i in range(NoOfRat):
        ax.plot(x_axis[i,:], BW_rate_original[i,:])
        
    plt.title('Simulated shifted backward-looking forward rates')
    plt.ylabel('Rate (%)')
    plt.xlabel('Time (years)')
    plt.xticks(locs,labels, rotation='45') 
    plt.grid()

    "Validating caplet prices"
    Notional = 1000000                                                                 # Notional
    K = -0.05                                                                          # Strike price
    M = 10000                                                                         # Number of Monte Carlo simulations

    [capLS, se_LS] = cap_price_MC(Notional, tau, K, V, L, NoOfSteps, NoOfRat, T, M, eul_steps, cap_shift, data.DF)         # Obtain the simulated caplet prices
    capBlack       = cap_price_Black(Notional, tau, T, K, V, L, cap_shift, data.DF)   
    
    # Generate nice ouput
    colname = list()
    for i in range(NoOfRat):
        name = '{}{}'.format('Cap', '(T_' + str(i+1) + ',' + 'T_' + str(i+2) + ')')
        colname.append(name)
  
    table = zip(colname, capBlack, capLS, 100 * abs(capLS - capBlack) /  capBlack, se_LS)
    header = ['Analytical Price', 'Simulated price', 'Error (%)', 'Standard\n error']  
    
    print(tabulate(table, headers = header, tablefmt="fancy_grid"))
        
    "Make density plot"
    density_plot(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, cap_shift)
      
# Time the program           
start = timeit.default_timer()   
print('Starting the function') 
mainCalculation()
stop = timeit.default_timer() 
print('Time: ', stop - start, '\n')