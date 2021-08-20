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
import sys


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
    
    Z     = np.random.normal(0, 1, [NoOfRates, NoOfSteps])                              # Create standard normal variables
    Zanti = -Z                                                                      # Antithetic variables
    W     = np.zeros([NoOfRates, NoOfSteps+1])                                          # Initialize Brownian motion
    Wanti = np.zeros([NoOfRates, NoOfSteps+1])                                      # Initialize Brownian motion

    C = insCorr                                                                     # Obtain correlation structure for the Brownian motions
    L = np.linalg.cholesky(C)                                                       # Apply Cholesky decomposition

    Z     = L @ Z
    Zanti = L @ Zanti


    for i in range(NoOfSteps):
        # Calculate the BM for every forward rate per time step i       
        W[:, i+1]     = W[:, i] + (np.power(dt, 0.5) * Z[:, i])
        Wanti[:, i+1] = Wanti[:, i] + (np.power(dt, 0.5) * Zanti[:, i])

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
        
    BWR_org        = np.zeros([NoOfRat, NoOfSteps + 1])                                     # Create matrices for the backward/forward rates
    BWR_org[:,0]   = np.transpose(L0_original)                                         # Set initial values
    BWR_shift      = np.zeros([NoOfRat, NoOfSteps + 1])                                   # Create matrices for the backward/forward rates
    BWR_shift[:,0] = np.transpose(L0_shifted)                                        # Set initial values
        
    BWR_org_anti        = np.zeros([NoOfRat, NoOfSteps + 1])                                # Do the same for the antithetic variables
    BWR_org_anti[:,0]   = np.transpose(L0_original) 
    BWR_shift_anti      = np.zeros([NoOfRat, NoOfSteps + 1]) 
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
           
        # Note that here we apply the risk-neutral dynamics
        matrix1 = np.tril(drift_approx[:,time:],-(time-1))                         # Help matrix
        matrix2 = np.tril(drift_approx_anti[:,time:],-(time-1))                    # Help matrix

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
        
        # Again risk-neutral dynamics
        matrix1 = np.tril(drift_approx[:,time:],-(time-1))
        matrix2 = np.tril(drift_approx_anti[:,time:],-(time-1))    

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

    # Define values after the rset date of forward rates
    BW_rate_original[BW_rate_original == 0] = BW_rate_anti_original[BW_rate_anti_original == 0] = np.nan
    BW_rate_shifted[BW_rate_shifted == 0]   = BW_rate_anti_shifted[BW_rate_anti_shifted == 0]   = np.nan

    return BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis 


# tau        = Difference between tenor points
# V          = Instantaneous volatility matrix of the large time step method
# L          = Initial forward rates
# NoOfSteps  = Number of steps used
# NoOfRates  = Number of rates that will be simulated
# T          = End date
# M          = Number of Monte Carlo simulations
# eul_steps  = Number of Euler discretization steps
# cap_shift  = SHift given by the market
def IRF_MC(tau, V, L, NoOfSteps, NoOfRates, T, M, eul_steps, cap_shift):
    """Calculate the price of a caplet using Monte Carlo simulation"""   
    ED_fut = np.zeros([NoOfRates,M])
    A = np.arange(NoOfRates).reshape((NoOfRates,1)) + 1               # Filter to get the correct rows and columns

    for i in range(M):
        [BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis] = Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, V, L, eul_steps, cap_shift)                # Obtain backward rates using a large time step
        # Obtain the futures rate using the martingale property
        ED_fut[:, i] = BW_rate_original[np.arange(A.shape[0])[:,None], (A)].reshape((NoOfRates))

    
    # Calculate standard errors
    se_Back  = np.sqrt(np.var(ED_fut, axis=1, ddof=1)) / np.sqrt(M)       
    
    return np.sum(ED_fut, axis=1) / M, se_Back


def mainCalculation():
    """Run the program to generate backward rates."""    
    tau         = 0.25                                                              # Difference between tenor points
    T           = 1                                                                 # Time horizon
    eul_steps   = 64                                                                # Euler discretization steps between two tenor points    
    
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

    "Futures rates"
    M = 100000                                                                        # Number of Monte Carlo simulations

    fut, fut_se = IRF_MC(tau, V, L, NoOfSteps, NoOfRat, T, M, eul_steps, cap_shift)

    # Generate nice ouput
    colname = list()
    for i in range(NoOfRat):
        name = '{}'.format('(T_{' + str(i+1) + '},' + 'T_{' + str(i+2) + '})')
        colname.append(name)
  
    table = zip(colname, L, fut, (fut - L), fut_se)
    header = ['Period', 'Forward rate', 'Futures rate', 'Convexity', 'Standard\n error']  
    
    print(tabulate(table, headers = header, tablefmt="latex_raw"))

# Time the program           
start = timeit.default_timer()   
print('Starting the function') 
mainCalculation()
stop = timeit.default_timer() 
print('Time: ', stop - start, '\n')