import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm
from random import randint
from tabulate import tabulate
import pandas as pd
import scipy.optimize as optimize


# Global variables
beta  = 0.9
rho   = 0
shift = 0.03

seed = 2500
# seed = randint(0, 100000)                                                             # Create random seed
np.random.seed(seed)     

# NoOfRates = Number of rates
def insCorr(NoOfRates):
    """Generate the instantaneous correlation matrix."""    
    N = NoOfRates                                                                     # N is the total number of rates
    C = np.ones((N, N))                                                               # Initialize the correlation matrix
    
    # Self-picked the value of 0.95
    np.fill_diagonal(C[:,1:], 0.95*np.ones(N-1))                                      # Set the value of the diagonal right from the true diagonal equal to 0.95  
    
    for i in range(N):
        for j in range(N):
            if j - i == 1:                                                            # Make the matrix symmetric
                C[j,i] = C[i,j]
            if j - i > 1:                                                             # Fill in the rest of the matrix
                k = i
                while k < j:
                    C[i,j] = C[i,j] * C[k,k+1]
                    C[j,i] = C[i,j]
                    k = k + 1   

    return C                                                                          # Return the correlation matrix

# NoOfRates      = Number of rates
# Initial_market = Initial market rates
def inFor(NoOfRates, initial_market): 
    """Generate the initial forward rates."""    
    L = initial_market[:NoOfRates].to_numpy()
    return L             

# T         = maturity time
# NoOfSteps = Number of steps between [0, T]
# NoOfRates = Number of rates that needs to be simulated\
# v         = Parameter for volatility
# dt        = step size
def GenerateBM_Vol(T, NoOfSteps, NoOfRates, v, dt):
    """"Generate Brownian motions."""
    Z      = np.random.normal(0, 1, [NoOfRates, NoOfSteps])                           # Create standard normal variables
    Zanti  = -Z                                                                       # Antithetic variables
       
    W     = np.zeros([NoOfRates, NoOfSteps+1])                                        # Initialize Brownian motion
    Wanti = np.zeros([NoOfRates, NoOfSteps+1])                                        # Initialize Brownian motion

    vol_BM     = np.zeros([NoOfSteps+1])                                              # Initialize Brownian motion
    vol_BManti = np.zeros([NoOfSteps+1])   

    C = insCorr(NoOfRates)                                                            # Obtain correlation structure for the Brownian motions
    L = np.linalg.cholesky(C)                                                         # Apply Cholesky decomposition

    Z    = L @ Z                                                                      # Correlate the BM's
    Zanti = L @ Zanti  
    
    Z_vol = np.random.normal(0, 1, NoOfSteps)
    Z_vol_anti = -Z_vol
    
    for i in range(NoOfSteps):
        # Calculate the BM for every forward rate per time step i       
        W[:, i+1]     = W[:, i] + (np.power(dt, 0.5) * Z[:, i])
        Wanti[:, i+1] = Wanti[:, i] + (np.power(dt, 0.5) * Zanti[:, i])
        
        vol_BM[i+1]     = vol_BM[i] + (np.power(dt, 0.5) * Z_vol[i])
        vol_BManti[i+1] = vol_BManti[i] + (np.power(dt, 0.5) * Z_vol_anti[i])
         
    V     = np.zeros([NoOfSteps+1])                                 
    Vanti = np.zeros([NoOfSteps+1]) 
    
    V[0]      = 1
    Vanti[0]  = 1

    for i in range(NoOfSteps):
        V[i+1] = V[i] + v * V[i] * (vol_BM[i+1] - vol_BM[i]) 
        Vanti[i+1] = Vanti[i] + v * Vanti[i] * (vol_BManti[i+1] - vol_BManti[i]) 

    return W, Wanti, V, Vanti

# NoOfSteps     = Number of time steps
# NoOfRat       = Number of rates
# T             = End time
# tau           = Difference between time steps
# L0            = Initial forward rates
# v             = Volatility parameter
# alpha         = Scale factor for vol process
def Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, L0, v, alpha):
    """Generate backward-looking forward rates"""
    # Initialize the backward rates
    eul_steps = int(NoOfSteps / (int(T / tau)))     

    BW_rate_original      = np.zeros([NoOfRates, NoOfSteps+1])                                 
    BW_rate_anti_original = np.zeros([NoOfRates, NoOfSteps+1]) 
    
    BW_rate_original[:,0]      = np.transpose(L0)
    BW_rate_anti_original[:,0] = np.transpose(L0)

    BW_rate_shift      = np.zeros([NoOfRates, NoOfSteps+1])                                 
    BW_rate_anti_shift = np.zeros([NoOfRates, NoOfSteps+1]) 
    
    BW_rate_shift[:,0]      = np.transpose(L0) + shift
    BW_rate_anti_shift[:,0] = np.transpose(L0) + shift
    
        
    # Euler discretization size
    dt = tau / eul_steps
    # Obtain the different parameters
    IC                   = insCorr(NoOfRates)
    [W, Wanti, V, Vanti] = GenerateBM_Vol(T, NoOfSteps, NoOfRates, v, dt)

    x_axis      = np.zeros([NoOfRates, NoOfSteps + 1])                                  # Values for x-axis
    Tj          = x_axis[:, 0] + (tau * np.linspace(1, NoOfRates, NoOfRates))           # Values for maturity

    for steps in range(NoOfSteps):                                                      # Loop per Euler step
        x_axis[:,steps+1] = x_axis[:, steps] + (tau / eul_steps)
        t                 = x_axis[:, steps]
        # Apply gamma function
        z1                         = Tj - t
        z1[z1 < 0]                 = 0
        gamma_rate                 = z1 / (Tj - (Tj-tau))
        gamma_rate[gamma_rate > 1] = 1
        
        X = Xanti = 0                                                           # Initialize drift term
        
        # Calculate drift terms
        Y = (tau * IC * alpha * V[steps] * (BW_rate_shift[:,steps] ** (beta)) * gamma_rate)       
        Z = 1 + tau * BW_rate_original[:,steps]
        X = Y / Z
               
        Yanti = (tau * IC * alpha * Vanti[steps] * (BW_rate_anti_shift[:,steps] ** (beta)) * gamma_rate)
        Zanti = 1 + tau * BW_rate_anti_original[:,steps]
        Xanti = Yanti / Zanti
                
        time = int(np.ceil(round(t[0],2) / tau))                                # Current time

        matrix1 = np.tril(X[:,time:],-time)                                     # Help matrix
        matrix2 = np.tril(Xanti[:,time:],-time)                                 # Help matrix       

        X      = np.nansum(matrix1, axis=1)                                     # Drift approximation
        Xanti  = np.nansum(matrix2, axis=1)                                     # Drift approximation

        
        # Obtain next value for backward-looking forward rate 
        BW_rate_shift[:,steps+1] = BW_rate_shift[:,steps] * np.exp((X * BW_rate_shift[:,steps] ** (beta-1) * alpha * V[steps] * gamma_rate  - 0.5 * (alpha * V[steps]) ** 2 * gamma_rate ** 2 * BW_rate_shift[:,steps] ** (2*beta-2)) * dt + alpha * V[steps] * gamma_rate * BW_rate_shift[:,steps] ** (beta-1) * (W[:,steps+1]-W[:,steps]))
        BW_rate_anti_shift[:,steps+1] = BW_rate_anti_shift[:,steps] * np.exp((Xanti * BW_rate_anti_shift[:,steps] ** (beta-1) * alpha * Vanti[steps] * gamma_rate  - 0.5 * (alpha * Vanti[steps]) ** 2 * gamma_rate ** 2 * BW_rate_anti_shift[:,steps] ** (2*beta-2)) * dt + alpha * Vanti[steps] * gamma_rate * BW_rate_anti_shift[:,steps] ** (beta-1) * (Wanti[:,steps+1]-Wanti[:,steps]))

        "Reflection principle"
        # BW_rate_shift[:,steps+1] = np.abs(BW_rate_shift[:,steps] + alpha * V[steps] * gamma_rate * (BW_rate_shift[:,steps] ** beta) * X * dt + \
            # alpha * V[steps] * (BW_rate_shift[:,steps] ** beta) * gamma_rate * (W[:,steps+1] - W[:,steps]))
            
        # BW_rate_anti_shift[:,steps+1] = np.abs(BW_rate_anti_shift[:,steps] + alpha * Vanti[steps] * gamma_rate * (BW_rate_anti_shift[:,steps] ** beta) * Xanti * dt + \
            # alpha * Vanti[steps] * (BW_rate_anti_shift[:,steps] ** beta) * gamma_rate * (Wanti[:,steps+1] - Wanti[:,steps]))

        BW_rate_original[:, steps+1] = BW_rate_shift[:, steps+1] - shift
        BW_rate_anti_original[:, steps+1] = BW_rate_anti_shift[:, steps+1] - shift

    # Define values after the accrual period
    hv = Tj / (tau / eul_steps)
    for i in range(NoOfRates-1):
        c = int(hv[i] + 1)
        BW_rate_shift[i, c:NoOfSteps+1+eul_steps]    = BW_rate_anti_shift[i, c:NoOfSteps+1+eul_steps]     = np.nan
        BW_rate_original[i, c:NoOfSteps+1+eul_steps] = BW_rate_anti_original[i,c:NoOfSteps+1+eul_steps] = np.nan

    return BW_rate_original, BW_rate_anti_original, BW_rate_shift, BW_rate_anti_shift, x_axis

# Notional   = Notional of the contract
# dif        = Difference between tenor points
# T          = Reset date of the caplet
# strike     = Strike price of the caplet
# L          = Initional forward rates
# v          = Volatility parameter
# IV         = Instantaneous volatility
def cap_price_Black_SABR(Notional, dif, T, strike, L, v, IV):  
    """Calculate the price of a caplet using Black's formula"""
    num_of_caps = int(T/dif)                                                                    # Number of caplets we want to validate
    cap_price   = np.zeros(num_of_caps)                                                         # Save caplet prices
    q           = 1              
    vsqr = 0
    
    for resetT in range(1, num_of_caps):                                                        # Apply Black's formula for every caplet                   
        # Specific caplet parameters
        tau_0     = resetT * dif
        tau_1     = tau_0 + dif
        R         = L[resetT] + shift
        alpha_cap = IV[resetT]
        K         = strike[resetT] + shift
        """Calculate effective SABR parameters"""
        tau = 2 * q * tau_0 + tau_1
        
        # Gamma
        a1  = 3 * q * rho ** 2 * (tau_1 - tau_0) ** 2
        a2  = (3 * tau ** 2 - tau_1 ** 2 + 5 * q * tau_0 ** 2 + 4 * tau_0 * tau_1)
        a3  = ((4 * q + 3) * (3 * q + 2) ** 2)         
        hv  = a1 * (a2 / a3)
             
        b1  = (2 * tau ** 3 + tau_1 ** 3 + (4 * q ** 2 - 2 * q) * tau_0 ** 3 + 6 * q * tau_0 ** 2 * tau_1)
        b2  = (4 * q + 3) * (2 * q + 1)
        gam = tau * (b1 / b2) + hv
            
        # v-square
        v_hat = v ** 2 * gam * ((2 * q + 1) / (tau ** 3 * tau_1))
          
        # H
        H = v ** 2 * ((tau ** 2 + 2 * q * tau_0 ** 2 + tau_1**2) / (2 * tau_1 * tau * (q+1))) - v_hat
        
        # alpha hat square
        alpha_hat = ((alpha_cap ** 2) / (2 * q + 1)) * (tau / tau_1) * np.exp(0.5 * H * tau_1)
        
        # rho hat square        
        rho_hat = rho * ((3 * tau ** 2 + 2 * q * tau_0 ** 2 + tau_1 ** 2) / (np.sqrt(gam) * (6 * q + 4)))
        
        # Black's equation
        x   = np.log(R / K)
        
        "OTM / ITM"
        # z   = np.sqrt(v_hat) / np.sqrt(alpha_hat) * (R * K) ** ((1-beta) / 2) * x
        # ksi = np.log((np.sqrt(1 - 2 * rho_hat * z + z ** 2) + z - rho_hat) / (1 - rho_hat))
        
        # # sigma-hagan               
        # a    = np.sqrt(alpha_hat) / (R * K) ** ((1 - beta) / 2)
        # b    = (1 + (1 - beta) ** 2 / 24 * x ** 2 + (1 - beta) ** 4 / 1920) ** -1
        # c    = z / ksi
        # d    = ((1 - beta) ** 2 / 24) * (alpha_hat / (R * K) ** (1 - beta))
        # e    = 0.25 * (rho_hat * beta * np.sqrt(v_hat) * np.sqrt(alpha_hat)) / (R * K) ** ((1 - beta) / 2)
        # f    = v_hat * ((2 - 3 * rho_hat ** 2) / 24)  
        # vsqr = (a * b * c * (1 + (d + e + f) * tau_1)) ** 2
        
        "ATM"
        a = np.sqrt(alpha_hat) / (R ** (1 - beta))
        b = ((1 - beta) ** 2 / 24) * (alpha_hat / (R ** (2 - 2 * beta)))
        c = 0.25 * ((rho_hat * beta * np.sqrt(alpha_hat) * np.sqrt(v_hat)) / (R ** (1 - beta)))
        d = ((2 - 3 * rho_hat ** 2) / 24) * v_hat
        vsqr = (a * (1 + (b + c + d) * tau_1)) ** 2     
        
        
        d1 = (x + (vsqr / 2) * tau_1) / (np.sqrt(vsqr)  * np.sqrt(tau_1)) 
        d2 = d1 - (np.sqrt(vsqr) * np.sqrt(tau_1))    

        P = 1 / (1 + dif * (-0.0049754))
        for i in range(0,resetT):
            P = P * (1 / (1 + dif * (L[i+1])))
                    
        cap_price[resetT] = Notional * P * dif * (R * norm.cdf(d1) - K * norm.cdf(d2) )
        if cap_price[resetT] / Notional < 10 ** -8:                                             # I take this as such a small value that we assume that the capprice is equal to zero
            cap_price[resetT] = 0
    
    return cap_price

# N          = Notional
# tau        = Difference between tenor points
# strike     = Strike price of the caplet
# L          = Initial forward rates
# NoOfSteps  = Number of steps used
# NoOfRates  = Number of rates that will be simulated
# T          = End date
# M          = Number of Monte Carlo simulations
# eul_steps  = Number of Euler discretization steps
# v          = Volatility parameter
# IV         = Instantaneous volatility
def cap_price_MC(N, tau, strike, L, NoOfSteps, NoOfRates, T, M, eul_steps, v, IV):
    """Calculate the price of a caplet using Monte Carlo simulation"""   
    cappriceBack = np.zeros([NoOfRates,M])
    discount_rate = np.zeros([NoOfRates, M])                                                    # Simulated ZCB
    discount_rate_anti = np.zeros([NoOfRates, M])                                               # Simulated ZCB

    K = strike + shift
    Tj  = (tau * np.linspace(1, NoOfRates, NoOfRates))                                                                       # Maturity date
    hv  = (Tj / (T / (NoOfSteps))).astype(int).tolist()                                                               # Help value 1
    for i in range(M):
        [BW, BW_anti, BW_shift, BW_anti_shift, x_axis] = Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, L, v, IV)                                  # Obtain backward rates using a large time step
        # Calculate payoffs and set negative values to zero (=max operator)  
        payoffLS                         = (BW_shift[np.arange(len(BW)), hv] - K) * N * tau
        payoffLS[payoffLS < 0]           = 0  
        payoffLS_anti                    = (BW_anti_shift[np.arange(len(BW_anti)), hv] - K) * N * tau
        payoffLS_anti[payoffLS_anti < 0] = 0
        
        for k in range(NoOfRates):                                                         # Loop to cover _valiple ZCB       
            # Obtain the rate using Euler discretization
            pSim = pSimanti = (1 + tau * (-0.0049754 )) 
            for j in range(0, k):
                pSim     = pSim * (1 + tau * BW[j+1, ((j+1) * eul_steps)])
                pSimanti = pSimanti * (1 + tau * BW_anti[j+1, ((j+1) * eul_steps)])

            discount_rate[k, i]      = pSim 
            discount_rate_anti[k, i] = pSimanti

        cappriceBack[:,i] = (np.reshape(np.divide(payoffLS , (discount_rate[:,i])),(NoOfRates,)) + np.reshape(np.divide(payoffLS_anti , (discount_rate_anti[:,i])),(NoOfRates,))) * 0.5

    # Calculate standard errors
    se_Back  = np.sqrt(np.var(cappriceBack, axis=1, ddof=1)) / np.sqrt(M)       
    
    return np.sum(cappriceBack, axis=1) / M, se_Back

# tau        = Difference between tenor points
# L          = Initial forward rates
# NoOfSteps  = Number of steps used
# NoOfRat    = Number of rates that will be simulated
# T          = End date
# M          = Number of Monte Carlo simulations
# eul_steps  = Number of Euler discretization steps
# v          = Volatility parameter
# IV         = Instantaneous volatility
def zero_coupon_bond(tau, L_un, NoOfSteps, NoOfRat, T, M, eul_steps, v, IV):
    """Calculate the ZCB rate analytically and using Monte Carlo simulation"""
    discount_rate = np.zeros([NoOfRat, M])                                               # Simulated ZCB
    analyt        = np.zeros(NoOfRat)

    L = L_un
    
    analyt[0] = 1 / (1 + tau * (-0.0049754))
    for i in range(NoOfRat-1):
        analyt[i+1] = analyt[i] * (1 / (1 + tau * L[i+1]))
        
    # Perform Monte Carlo Simulation
    for i in range(M):
        [BW, BW_anti, BW_shift, BW_anti_shift, x_axis] = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, L_un, v, IV)                                  # Obtain backward rates using a large time step
        
        for k in range(NoOfRat):                                                         # Loop to cover _valiple ZCB       
            # Obtain the rate using Euler discretization
            pSim = pSimanti = (1 + tau * (-0.0049754)) 
            for j in range(0, k):
                pSim = pSim * (1 + tau * BW[j+1, ((j+1) * eul_steps)])
                pSimanti = pSimanti * (1 + tau * BW_anti[j+1, ((j+1) * eul_steps)])
            discount_rate[k, i] = (1 / pSim + 1 / pSimanti) * 0.5
            
    # Calculate standard errors
    se_Sim = np.sqrt(np.var(discount_rate, axis=1, ddof=1)) / np.sqrt(M)
    
    # [Analytical rate, Rate using Euler, Rate using large steps]
    return analyt, np.sum(discount_rate, axis=1) / M, se_Sim


def mainCalculation():
    """Run the program to generate backward rates."""    
    tau         = 0.25                                                              # Difference between tenor points
    T           = 1                                                                 # Time horizon
    eul_steps   = 64                                                                # Euler discretization steps between two tenor points
      
    NoOfSteps   = int(T / tau) * eul_steps                                          # Number of time-steps                                                    
    NoOfRat     = int(T / tau)                                                      # Number of backward rates we want to generate
      
    v           = 0.1                                                         # Volatility parameter

    # Obtain the correct data (Initial rates, Instantaneous volatilities)
    df = pd.read_excel (r'C:\Users\hackt\Documents\Thesis Rabobank\Python\Caplets_3Months_Test.xlsx')

    help1 = df.iloc[16:,:]
    data = help1.iloc[:,[1,2,5,7]]
    data.reset_index(drop=True, inplace=True)
    data.columns = ['Date', 'CapletForward', 'DF', 'IV']

    alpha = np.zeros(NoOfRat)
    for i in range(len(alpha)):
        alpha[i] = data.IV[i]

    L         = inFor(NoOfRat, data.CapletForward)  / 100                            # Initial rates
    BW_rate, BW_rate_anti, BW_shift, BW_anti_shift, x_axis_fast = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, L, v, alpha) 
      
    "Plotting the rates"
    # Obtain labels and locations for the x-axis
    labels = np.zeros(NoOfRat+1) ; locs = np.zeros(NoOfRat+1) 
    hv = 0
    for i in range(NoOfRat+1):
        labels[i] = hv
        locs[i] = i * tau
        hv += 0.25
      
    fig, ax = plt.subplots()
    for i in range(NoOfRat):
        ax.plot(x_axis_fast[i,:], BW_rate[i,:])
        
    plt.title('Simulated shifted backward-looking forward rates')
    plt.ylabel('Rate (%)')
    plt.xlabel('Time (years)')
    plt.xticks(locs,labels, rotation='45') 
    plt.grid() 
    

    "Validate the model by pricing caplets"        
    N = 10000                                                                         # Notional
    M = 500                                                                       # Number of Monte Carlo simulations
    K = L                                                                             # ATM strikes

    [capLS, se_LS]= cap_price_MC(N, tau, K, L, NoOfSteps, NoOfRat, T, M, eul_steps, v, alpha)      # Simulated prices
    capBlack = cap_price_Black_SABR(N, tau, T, K, L, v, data.IV)                                # Analytical prices

    colname = list()
    for i in range(NoOfRat):
        name = '{}{}'.format('Cap', '(T_{' + str(i) + '},' + 'T_{' + str(i+1) + '})')
        colname.append(name)
  
    table2 = zip(colname, capBlack, capLS, 100 * abs(capLS - capBlack) /  capBlack, se_LS)
    header2 = ['Analytical Price', 'Simulated price', 'Error (%)', 'Standard\n error']  
    
    print(tabulate(table2, headers = header2, tablefmt="fancy_grid"))
    
    "Valiating zero-coupon bond prices"
    M = 500
    
    [anrate, sim_rate, se_Sim] = zero_coupon_bond(tau, L, NoOfSteps, NoOfRat, T, M, eul_steps, v, alpha)      # Obtain the differenct calculatd zero-coupon rates
    
    
    colname = list()
    for i in range(NoOfRat):
        name = '{}{}'.format('P', '(T_{' + str(i+1) + '})')
        colname.append(name)
  
    table = zip(colname, anrate, sim_rate, 100 * abs(sim_rate - anrate) / anrate, se_Sim)
    header = ['Analytical Price', 'Simulated price', 'Error (%)', 'Standard error']
    
    print(tabulate(table, headers = header, tablefmt="fancy_grid"))

start = timeit.default_timer()   
print('Starting the plotting function') 
mainCalculation()
stop = timeit.default_timer() 
print('Time: ', stop - start, '\n')