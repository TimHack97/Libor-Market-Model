import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm
from random import randint

seed = 1
# seed = randint(0, 100000)

np.random.seed(seed)                                                                  # Set seed

# T         = Maturity  time
# NoOfSteps = Number of steps in the time grid
# NoOfFor   = Number of forward rates
# Z         = Matrix containing standard normal random variables
def GenerateBM(T, NoOfSteps, NoOfFor, insCorr):
    """"Generate Brownian motions."""
    dt = T / NoOfSteps                                                              # Timegrid
    
    Z = np.random.normal(0, 1, [NoOfFor, NoOfSteps])                                # Create standard normal variables
    Zanti = -Z                                                                      # Antithetic variables
    
    W = np.zeros([NoOfFor, NoOfSteps+1])                                            # Initialize Brownian motion
    Wanti = np.zeros([NoOfFor, NoOfSteps+1])

    C = insCorr                                                                     # Obtain correlation structure for the Brownian motions
    L = np.linalg.cholesky(C)                                                       # Apply Cholesky decomposition

    for i in range(NoOfSteps):
        # Calculate the BM for every forward rate per time step i       
        W[:, i+1] = W[:, i] + (np.power(dt, 0.5) * Z[:, i])
        Wanti[:, i+1] = Wanti[:, i] + (np.power(dt, 0.5) * Zanti[:, i])

    W = L @ W                                                                       # Correlate the BM's
    Wanti = L @ Wanti

    return W, Wanti                                                                 # Return the generated Brownian motion

# NoOfSteps = Number of steps in the time grid
# NoOfFor   = Number of forward rates
def insVol(NoOfSteps, NoOfFor): 
    """Generate the instantaneous volatility matrix."""
    np.random.seed(seed)
    Vinit = np.random.uniform(0.2, 0.4, NoOfFor)                                    # Draw from a uniform distribution every IV for every forward rate
    V = np.zeros([NoOfFor, NoOfSteps]) 
       
    tenor = NoOfSteps / NoOfFor                                                     # Tenor grid
    
    for i in range(NoOfSteps):                                                      # Loop per time step
        for j in range(NoOfFor):                                                    # Loop per forward rate
            if i == 0:
                V[j,:] = Vinit[j]                                                   # Set every time step equal to the initialized IV
            if i >= tenor*j:
                V[0:j,i:] = np.nan                                                  # If the forward rate is 'dead' reset the value to NaN

    return V                                                                        # Return the instantaneous volatility matrix

# NoOfFor = Number of forward rates
def insCorr(NoOfFor):
    """Generate the instantaneous correlation matrix."""    
    N = NoOfFor                                                                     # N is the total number of forward rates
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

# NoOfFor = Number of forward rates
def inFor(NoOfFor): 
    """Generate the initial forward rates."""        
    L = np.random.uniform(0.02, 0.08, NoOfFor)                                      # Draw from an uniform distribution between [0.2,0.8]
    L = np.sort(L)                                                                  # Sort the values such that the shortest forward rate has the lowest value

    return L                                                                        # Return the forward rates

# NoOfSteps = Number of steps in the time grid
# NoOfFor   = Number of forward rates
# T         = Maturity time
# tau       = Difference between T_i+1 and T_i
# V         = Instantaneous volatility matrix
# L0        = Initial forward rates
def GenerateFRWEul(NoOfSteps, NoOfFor, T, tau, V, L0):
    """Generate the forward rates using Euler discretization."""   
    dt = T / NoOfSteps                                                              # Time-grid
    
    # Obtain the initial values needed for the Euler discretization
    C = insCorr(NoOfFor)                                                            # Instantaneous correlation matrix
    BM, BManti = GenerateBM(T, NoOfSteps, NoOfFor, C)                               # Generated Brownian motions
    
    # Initialzie the forward rates
    FRW = np.zeros([NoOfFor, NoOfSteps+1])                                          # NoOfSteps + 1 since we have time 0, 1, 2,.., N = N+1 steps
    FRW[:,0] = np.transpose(L0) 
    
    FRWanti = np.zeros([NoOfFor, NoOfSteps+1])                                      # NoOfSteps + 1 since we have time 0, 1, 2,.., N = N+1 steps
    FRWanti[:,0] = np.transpose(L0) 


    for time in range(NoOfSteps):                                                   # Loop per time step
        for k in range(0,NoOfFor):                                                  # Loop per forward rate
            q = np.int(np.ceil((time*NoOfFor) / NoOfSteps + 0.01))                  # Smallest tenor point that is larger than the current time step
            X = 0
            Xanti = 0
            if q <= (k+1):  
                for j in range(q-1,k+1):                                            # Perform the summation of the drift term
                    Y = tau * C[k,j] * V[j,time] * FRW[j,time]
                    Z = 1 + tau * FRW[j,time]
                    X += Y / Z
                    
                    Yanti = tau * C[k,j] * V[j,time] * FRWanti[j,time]
                    Zanti = 1 + tau * FRWanti[j,time]
                    Xanti += Yanti / Zanti

            FRW[k,time+1] = FRW[k,time] + V[k,time] * FRW[k,time] * X * dt + \
                V[k,time] * FRW[k,time] * (BM[k,time+1] - BM[k,time])                           # Perform the Euler discretization
    
            FRWanti[k,time+1] = FRWanti[k,time] + V[k,time] * FRWanti[k,time] * Xanti * dt + \
                V[k,time] * FRWanti[k,time] * (BManti[k,time+1] - BManti[k,time])       
    
    FRW = (FRW + FRWanti) / 2
    
    return FRW                                                                      # Return the forward rates

# V         = Instantaneous volatility matrix
# IC        = Instantaneous correlation matrix
# time      = Current time
# frwrate   = Forward rate
# sumrate   = Forward rate showing in the sum
# tau       = Size time-step
def getvalueC(V, IC, time, frwrate, sumrate, tau):   
    """Approximate the integral over the time-step from the instantaneous volatilties and correlation"""
    
    # Note that I can just use the first value of the matrices since the NaN 
    # is taken care of in the other function and the IV is constant
    fa = V[frwrate, 0] * V[sumrate, 0] * \
        IC[frwrate, sumrate]                                                    # Left part of the integral
    fb = V[frwrate, 0] * V[sumrate, 0] * \
        IC[frwrate, sumrate]                                                    # Right part of the integral
    
    Cval = tau * ((fa + fb) / 2)                                                # Trapezoidal approximation of an integral

    return Cval                                                                 # Return the approximated integral

# tau           = Difference between time steps
# FRW           = Matrix with the till time calculated forward rates
# bottom        = First value of the summation
# frwrate       = Number of the current forward rate
# V             = Instantaneous volatility matrix
# IC            = Instantaneous correlation matrix
# time          = Current time
# timeIntegral  = Time for the integral Cij, this is different from the time for v(T_k) when using the better drift approx.
def approxDrift(tau, FRW, bottom, frwrate, V, IC, time, timeIntegral):
    """Approximate the drift integral"""
    X = 0                                                                       # Approximation value
    for k in range(bottom, frwrate+1):
        sumrate = k                                                             # Current value of the summation of the drift
        Cij = getvalueC(V, IC, timeIntegral, frwrate, sumrate, tau)             # Calculate the C integral
        if time == timeIntegral:                                                # This means setting L(u) = L(k)
            X += (((tau * FRW[k,time]) /    \
                 (1 + tau * FRW[k,time])) * Cij)                                # Approximate the drift formula
        else:
            approxFRW = 0.5 * (FRW[k,time] + FRW[k,time-1])
            X += (((tau * approxFRW) /    \
                 (1 + tau * approxFRW)) * Cij)                                  # Approximate the drift formula            
    return X

# NoOfSteps     = Number of time steps
# NoOfFor       = Number of forward rates
# T             = End time
# tau           = Difference between time steps
# V             = Instantaneous volatility matrix
# L0            = Initial forward rates
def GenerateFRWLS(NoOfSteps, NoOfFor, T, tau, V, L0):
    """Generate forwward rates using a big time step."""
    # Obtain the initial values needed for the simulation
    IC = insCorr(NoOfFor)                                                            # Instantaneous correlation matrix
    BM, BManti = GenerateBM(T, NoOfSteps, NoOfFor, IC)                               # Generated Brownian motions
    
    # Initialize the forward rates
    FRW = np.zeros([NoOfFor, NoOfSteps+1]) 
    FRW[:,0] = np.transpose(L0) 
    
    FRWanti = np.zeros([NoOfFor, NoOfSteps+1]) 
    FRWanti[:,0] = np.transpose(L0)     
        
    for time in range(NoOfSteps):
        for frwrate in range(NoOfFor): 
            q = np.int(np.ceil((time*NoOfFor) / NoOfSteps + 0.01))                  # Smallest tenor point that is larger than the current time step
            Cii = getvalueC(V, IC, time, frwrate, frwrate, tau)
            if q <= (frwrate+1):
                X1 = approxDrift(tau, FRW, (q-1), frwrate, V, IC, time, time)       # q-1 is the bottom of the summation sign in drift
                Y = -0.5 * Cii
                Z = V[frwrate, 0] * (BM[frwrate , time+1] - BM[frwrate , time])

                FRW[frwrate,time+1] = FRW[frwrate,time] * np.exp(X1 + Y + Z)  

                # Perform better approximation (Predictor-Corrector)
                X2 = approxDrift(tau, FRW, (q-1), frwrate, V, IC, time+1, time)
                FRW[frwrate,time+1] = FRW[frwrate,time] * np.exp(((X1 + X2) / 2) + Y + Z)
                
                # Now using antithetic variables
                X1anti = approxDrift(tau, FRWanti, (q-1), frwrate, V, IC, time, time)             # q-1 is the bottom of the summation sign
                Yanti = -0.5 * Cii
                Zanti = V[frwrate, 0] * (BManti[frwrate , time+1] - BManti[frwrate , time])

                FRWanti[frwrate,time+1] = FRWanti[frwrate,time] * np.exp(X1anti + Yanti + Zanti)  

                # Perform better approximation (Predictor-Corrector)
                X2anti = approxDrift(tau, FRWanti, (q-1), frwrate, V, IC, time+1, time)
                FRWanti[frwrate,time+1] = FRWanti[frwrate,time] * np.exp(((X1anti + X2anti) / 2) + Yanti + Zanti)
            else:
                FRW[frwrate,time+1] = np.nan
                FRWanti[frwrate,time+1] = np.nan
    
    FRW = (FRW + FRWanti) / 2
    
    return FRW

# Notional  = Notional of the contract
# tau       = Difference between tenor points
# resetT    = Reset date of the caplet
# K         = Strike price of the caplet
# V         = Instantaneous volatility matrix
# L         = Initional forward rates
# spotrate  = Spotrate
def capPriceBlack(Notional, tau, resetT, K, V, L, spotrate):  
    """Calculate the price of a caplet using Black's formula"""
    vsqr = 0.5 * V[resetT-1, 0] * V[resetT-1, 0]
    
    d1 = (np.log(L[resetT-1] / K) + (vsqr / 2)) / (np.sqrt(vsqr)) 
    d2 = d1 - (np.sqrt(vsqr))

    P = (1 / (1 + tau * spotrate))
    for i in range(0,resetT):      
        # Loop to calculate the value of P(0,T_n)
        P = P * (1 / (1 + tau * L[i]) )
    
    return Notional * P * (L[resetT-1] * norm.cdf(d1) - K * norm.cdf(d2) )

# tau       = Difference between tenor points
# resetT    = Reset date of the caplet
# K         = Strike price of the caplet
# V         = Instantaneous volatility matrix of the large time step method
# V2        = Instantaneous volatility matrix of the Euler method
# L         = Initial forward rates
# NoOfSteps = Number of steps used
# NoOfFor = Number of forward rates that will be simulated
# T         = End date
# spotrate  = Spotrate
# M         = Number of Monte Carlo simulations
# EulSteps  = Number of Euler discretization steps
def capPriceMC(tau, resetT, K, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M, EulSteps):
    """Calculate the price of a caplet using Monte Carlo simulation"""
    cappriceEUL = np.zeros(M)                                                   # Save caplet prices for Euler method
    cappriceLS = np.zeros(M)                                                    # Save caplet prices for large step method
    
    for i in range(M):
        FRW = GenerateFRWEul(NoOfSteps*EulSteps, NoOfFor, T, tau, V2, L)        # Obtain forward rates using Euler discretization
        FRWLS = GenerateFRWLS(NoOfSteps, NoOfFor, T, tau, V, L)                 # Obtain forward rates using a large time step
    
        # Calculate the payoffs using the forward rates from Euler discretization
        payoff = max(FRW[resetT-1, resetT*EulSteps] - K, 0)
        discount = (1 + tau * spotrate)
        for j in range(0,resetT):
            discount = discount * (1 + tau * FRW[j,(j+1)*EulSteps])
        cappriceEUL[i] = payoff / discount
        
        # Calculate the payoffs using the forward rates from the large step method
        payoff2 = max(FRWLS[resetT-1, resetT] - K, 0)
        discount2 = (1 + tau * spotrate)
        for k in range(0,resetT):
            discount2 = discount2 * (1 + tau * FRWLS[k,k+1])
        cappriceLS[i] = payoff2 / discount2
    
    return np.sum(cappriceEUL) / M, np.sum(cappriceLS) / M

# tau       = Difference between tenor points
# Tn        = End point of the ZCB rate you want
# V         = Instantaneous volatility matrix of the large time step method
# V2        = Instantaneous volatility matrix of the Euler method
# L         = Initial forward rates
# NoOfSteps = Number of steps used
# NoOfFor = Number of forward rates that will be simulated
# T         = End date
# spotrate  = Spotrate
# M         = Number of Monte Carlo simulations
# EulSteps  = Number of Euler discretization steps
def discRate(tau, Tn, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M, EulSteps):
    """Calculate the ZCB rate analytically and using Monte Carlo simulation"""
    eurate = np.zeros(M)                                                        # Rate using Euler discretization
    lsrate = np.zeros(M)                                                        # Rate using a larger time step    
    
    # Value to calculate the rate analytically 
    helpval = (1 + tau * spotrate)
    for i in range(0,Tn-1):
        helpval = helpval * (1 + tau * L[i])  
    
    # Perform Monte Carlo Simulation
    for i in range(M):
        FRW = GenerateFRWEul(NoOfSteps*EulSteps, NoOfFor, T, tau, V2, L)        # Obtain forward rates using a Euler discretization
        FRWLS = GenerateFRWLS(NoOfSteps, NoOfFor, T, tau, V, L)                 # Obtain forward rates using a large time step
        
        # Obtain the rate using Euler discretization
        peul = (1 + tau * spotrate)
        for j in range(0,Tn-1):
            peul = peul * (1 + tau * FRW[j,(j+1)*EulSteps])
        eurate[i] = 1 / peul
        
        # Obtain the rate using large time step
        p = (1 + tau * spotrate)
        for j in range(0,Tn-1):
            p = p * (1 + tau * FRWLS[j,j+1])
        lsrate[i] = 1 / p
        
        
    # [Analytical rate, Rate using Euler, Rate using large steps]
    return 1 / helpval, np.sum(eurate) / M, np.sum(lsrate) / M

def mainCalculation():
    """Run the program to generate forward rates."""    
    tau       = 0.25                                                                # Difference between time steps
    T         = 5                                                                   # Time horizon
    EulSteps  = 1000                                                                # Euler discretization steps between two tenor points
    
    NoOfSteps = int(T / tau)                                                        # Number of time-steps                                                    
    NoOfFor = int(T / tau)                                                          # Number of forward rates we want to generate
    
    # Note that normally you will obtain these using calibration, now I simulate them randomly
    V = insVol(NoOfSteps, NoOfFor)
    V2 = insVol(NoOfSteps*EulSteps, NoOfFor)
    L = inFor(NoOfFor)
    
    FRW = GenerateFRWEul(NoOfSteps*EulSteps, NoOfFor, T, tau, V2, L)                    # Generate forward rates using Euler discretization

    # Obtain labels and locations for the x-axis
    labels = np.zeros(NoOfFor+1) ; locs = np.zeros(NoOfFor+1) ; locsLS = np.zeros(NoOfFor+1)
    hv = 0
    for i in range(NoOfFor+1):
        labels[i] = hv
        locs[i] = i * EulSteps
        locsLS[i] = i
        hv += 0.25

    # Plot the different forward rates
    plt.figure()
    for i in range(len(FRW)):
            plt.plot(FRW[i,:])
    plt.title('Forward rates using Euler discretization')
    plt.ylabel('Forward rate')
    plt.xlabel('Time (years)')
    plt.xticks(locs,labels, rotation='45')
    plt.show()
    
    FRWLS = GenerateFRWLS(NoOfSteps, NoOfFor, T, tau, V, L)                         # Generate forward rates using a large time-step
    
    # Plot the different forward rates
    plt.figure()
    for i in range(len(FRWLS)):
            plt.plot(FRWLS[i,:])
    plt.title('Forward rates using big time-steps')
    plt.ylabel('Forward rate')
    plt.xlabel('Time (years)')
    plt.xticks(locsLS, labels, rotation='45')
    plt.show()
 
    
def pricingCaplet():
    """Run the program to price caplets."""    
    tau       = 0.25                                                                # Difference between time steps
    T         = 1                                                                   # Time horizon
    M         = 1000                                                                # Number of Monte Carlo simulations
    EulSteps  = 100                                                                 # Euler discretization steps between two tenor points

    NoOfSteps = int(T / tau)                                                        # Number of time-steps                                                    
    NoOfFor   = int(T / tau)                                                        # Number of forward rates we want to generate
    
    # Note that normally you will obtain these using calibration, now I simulate them randomly
    V         = insVol(NoOfSteps, NoOfFor)
    V2        = insVol(NoOfSteps*EulSteps, NoOfFor)
    L         = inFor(NoOfFor)
    spotrate  = L[0] / 2
    
    [capEUL, capLS] = capPriceMC(tau, 3, 0.02, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M, EulSteps)         # Obtain the simulated caplet prices
    capBLACK = capPriceBlack(1, tau, 3, 0.02, V, L, spotrate)                                                  # Obtain the analytical caplet price
    
    print('Analytical price is: ', capBLACK, 'The Large step price is: ', capLS)
    # print('The absolute difference is: ', np.abs(capBLACK - capLS))
    print('The procentual difference is: ', 100 * abs(capLS - capBLACK) / capBLACK)

    print('Analytical price is: ', capBLACK, 'The Euler price is: ', capEUL)
    # print('The absolute difference is: ', np.abs(capBLACK - capEUL), '\n')
    print('The procentual difference is: ', 100 * abs(capEUL - capBLACK) / capBLACK, '\n')   
    
def generatingDiscountRates(): 
    """Run the program to generate discount rates."""    
    tau       = 0.25                                                                # Difference between time steps
    T         = 1                                                                   # Time horizon
    M         = 1000                                                                # Number of Monte Carlo simulations
    EulSteps  = 100                                                                 # Euler discretization steps

    NoOfSteps = int(T / tau)                                                        # Number of time-steps                                                    
    NoOfFor   = int(T / tau)                                                        # Number of forward rates we want to generate
    
    # Note that normally you will obtain these using calibration, now I simulate them randomly
    V         = insVol(NoOfSteps, NoOfFor)
    V2        = insVol(NoOfSteps*EulSteps, NoOfFor)
    L         = inFor(NoOfFor)
    spotrate  = L[0] / 2 
      
    [anrate, eurate, lsrate] = discRate(tau, 3, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M, EulSteps)      # Obtain the differenct calculatd zero-coupon rates
    
    print('Analytical rate is: ', anrate, 'The Large step rate is: ', lsrate)
    # print('The absolute difference is: ', np.abs(anrate - lsrate))
    print('The procentual difference is: ', 100 * abs(lsrate - anrate) / anrate)

    print('Analytical rate is: ', anrate, 'The Euler rate is: ', eurate)
    # print('The absolute difference is: ', np.abs(anrate - eurate), '\n')
    print('The procentual difference is: ', 100 * abs(eurate - anrate) / anrate, '\n')    
    

#%% Run the plot function
np.random.seed(seed)                                                                  # Set seed

start = timeit.default_timer()   
print('Starting the plotting function') 
mainCalculation()
stop = timeit.default_timer() 
print('Time: ', stop - start, '\n')

#%% Run the caplet pricing function
np.random.seed(seed)                                                                  # Set seed

start = timeit.default_timer()   
print('Starting the caplet pricing function') 
pricingCaplet()
stop = timeit.default_timer() 
print('Time: ', stop - start, '\n')

#%% Run the discount rate function
np.random.seed(seed)                                                                  # Set seed

start = timeit.default_timer()
print('Starting the discount rate generating function')    
generatingDiscountRates()
stop = timeit.default_timer() 
print('Time: ', stop - start)