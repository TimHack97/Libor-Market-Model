import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm
from random import randint
from tabulate import tabulate

seed = 10
# seed = randint(0, 100000)                                                           # Create random seed

np.random.seed(seed)                                                                  # Set seed

# T         = End  time
# NoOfSteps = Number of steps in the time grid
# NoOfFor   = Number of forward rates
# insCorr   = Matrix containing instantaneous correlation
def GenerateBM(T, NoOfSteps, NoOfFor, insCorr):
    """"Generate Brownian motions."""
    dt = T / NoOfSteps                                                              # Discretization grid
    
    Z     = np.random.normal(0, 1, [NoOfFor, NoOfSteps])                                # Create standard normal variables
    Zanti = -Z                                                                      # Antithetic variables
    
    W     = np.zeros([NoOfFor, NoOfSteps+1])                                            # Initialize Brownian motion
    Wanti = np.zeros([NoOfFor, NoOfSteps+1])                                        # Initialize Brownian motion

    C = insCorr                                                                     # Obtain correlation structure for the Brownian motions
    L = np.linalg.cholesky(C)                                                       # Apply Cholesky decomposition

    # Perform Cholesky decomposition
    Z     = L @ Z
    Zanti = L @ Zanti

    for i in range(NoOfSteps):
        # Calculate the BM for every forward rate per time step i       
        W[:, i+1] = W[:, i] + (np.power(dt, 0.5) * Z[:, i])
        Wanti[:, i+1] = Wanti[:, i] + (np.power(dt, 0.5) * Zanti[:, i])

    return W, Wanti                                                                 # Return the generated Brownian motion

# NoOfSteps = Number of steps in the time grid
# NoOfFor   = Number of forward rates
def insVol(NoOfSteps, NoOfFor): 
    """Generate the instantaneous volatility matrix."""
    np.random.seed(seed)
    Vinit = np.random.uniform(0.1, 0.2, NoOfFor)                                    # Draw from a uniform distribution every IV for every forward rate
    Vinit = np.sort(Vinit)
    V = np.zeros([NoOfFor, NoOfSteps]) 
       
    tenor_steps = NoOfSteps / NoOfFor                                               # Steps in between tenor points
    
    for i in range(NoOfSteps):                                                      # Loop per time step
        for j in range(NoOfFor):                                                    # Loop per forward rate
            if i == 0:
                V[j,:] = Vinit[j]                                                   # Set every time step equal to the initialized IV
            if i >= tenor_steps*j:
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
def Generate_FRW_Eul(NoOfSteps, NoOfFor, T, tau, V, L0):
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
            X = Xanti =  0
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

    return FRW, FRWanti                                                             # Return the forward rates

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
        else:                                                                   # Predictor-Corrector approximation
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
def Generate_FRW_LS(NoOfSteps, NoOfFor, T, tau, V, L0):
    """Generate forwward rates using a big time step."""
    # Obtain the initial values needed for the simulation
    IC         = insCorr(NoOfFor)                                                            # Instantaneous correlation matrix
    BM, BManti = GenerateBM(T, NoOfSteps, NoOfFor, IC)                               # Generated Brownian motions
    
    # Initialize the forward rates
    FRW      = np.zeros([NoOfFor, NoOfSteps+1]) 
    FRW[:,0] = np.transpose(L0) 
    
    FRWanti      = np.zeros([NoOfFor, NoOfSteps+1]) 
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
                FRW[frwrate,time+1] = FRW[frwrate,time] * np.exp(X2 + Y + Z)
                
                # Now using antithetic variables
                X1anti = approxDrift(tau, FRWanti, (q-1), frwrate, V, IC, time, time)             # q-1 is the bottom of the summation sign
                Yanti = -0.5 * Cii
                Zanti = V[frwrate, 0] * (BManti[frwrate , time+1] - BManti[frwrate , time])

                FRWanti[frwrate,time+1] = FRWanti[frwrate,time] * np.exp(X1anti + Yanti + Zanti)  

                # Perform better approximation (Predictor-Corrector)
                X2anti = approxDrift(tau, FRWanti, (q-1), frwrate, V, IC, time+1, time)
                FRWanti[frwrate,time+1] = FRWanti[frwrate,time] * np.exp(X2anti + Yanti + Zanti)
            else:
                FRW[frwrate,time+1] = np.nan
                FRWanti[frwrate,time+1] = np.nan
    
    return FRW, FRWanti                                                                         # Return forward rates

# Notional  = Notional of the contract
# tau       = Difference between tenor points
# T         = Reset date of the caplet
# Strike    = Strike price of the caplet
# V         = Instantaneous volatility matrix
# L         = Initional forward rates
# spotrate  = Spotrate
def cap_price_Black(Notional, tau, T, strike, V, L, spotrate):  
    """Calculate the price of a caplet using Black's formula"""
    num_of_caps = int(T/tau)                                                                    # Number of caplets we want to calculate
    cap_price = np.zeros(num_of_caps)                                                           # Save caplet prices
    
    for resetT in range(1, num_of_caps+1):                                                      # Apply Black's formula for every caplet            
        K = strike[resetT-1]   
    
        vsqr = (resetT * tau) * V[resetT-1, 0] * V[resetT-1, 0]
        
        d1 = (np.log(L[resetT-1] / K) + (vsqr / 2)) / (np.sqrt(vsqr)) 
        d2 = d1 - (np.sqrt(vsqr))
    
        P = (1 / (1 + tau * spotrate))
        for i in range(0,resetT):      
            # Loop to calculate the value of P(0,T_n)
            P = P * (1 / (1 + tau * L[i]) )
            
        cap_price[resetT-1] = Notional * P * (L[resetT-1] * norm.cdf(d1) - K * norm.cdf(d2) )

    return cap_price

# N         = Notional
# tau       = Difference between tenor points
# resetT    = Reset date of the caplet
# strike    = Strike price of the caplet
# V         = Instantaneous volatility matrix of the large time step method
# V2        = Instantaneous volatility matrix of the Euler method
# L         = Initial forward rates
# NoOfSteps = Number of steps used
# NoOfFor = Number of forward rates that will be simulated
# T         = End date
# spotrate  = Spotrate
# M         = Number of Monte Carlo simulations
# EulSteps  = Number of Euler discretization steps
def cap_price_MC(N, tau, strike, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M, EulSteps):
    """Calculate the price of a caplet using Monte Carlo simulation"""
    cappriceEUL = np.zeros([NoOfFor, M])                                      # Save caplet prices for Euler method
    cappriceLS  = np.zeros([NoOfFor, M])                                       # Save caplet prices for large step method
    
    for i in range(M):
        FRW, FRWanti = Generate_FRW_Eul(NoOfSteps*EulSteps, NoOfFor, T, tau, V2, L)          # Obtain forward rates using Euler discretization
        FRWLS, FRWLSanti = Generate_FRW_LS(NoOfSteps, NoOfFor, T, tau, V, L)                 # Obtain forward rates using a large time step
        for resetT in range(1, NoOfFor+1):                                                   # Calculate every caplet price
            # Calculate the payoffs using the forward rates from Euler discretization
            K = strike[resetT-1]
            
            payoff = N * max(FRW[resetT-1, resetT*EulSteps] - K, 0)
            payoffAnti = N * max(FRWanti[resetT-1, resetT*EulSteps] - K, 0)
            discount = discountAnti = (1 + tau * spotrate)
            for j in range(0,resetT):
                discount = discount * (1 + tau * FRW[j,(j+1)*EulSteps])
                discountAnti = discountAnti * (1 + tau * FRWanti[j,(j+1)*EulSteps])
            cappriceEUL[resetT-1, i] = (payoff / discount + payoffAnti / discountAnti) * 0.5
            
            # Calculate the payoffs using the forward rates from the large step method
            payoffLS = N * max(FRWLS[resetT-1, resetT] - K, 0)
            payoffLSanti = N * max(FRWLSanti[resetT-1, resetT] - K, 0)
            discountLS = discountLSanti = (1 + tau * spotrate)
            for k in range(0,resetT):
                discountLS = discountLS * (1 + tau * FRWLS[k,k+1])
                discountLSanti = discountLSanti * (1 + tau * FRWLSanti[k,k+1])
            cappriceLS[resetT-1, i] = (payoffLS / discountLS + payoffLSanti / discountLSanti) * 0.5
    
    # Calculate standard errors
    se_Eul = np.sqrt(np.var(cappriceEUL, axis=1, ddof=1)) / np.sqrt(M)
    se_LS  = np.sqrt(np.var(cappriceLS, axis=1, ddof=1)) / np.sqrt(M)       
    
    return np.sum(cappriceEUL, axis=1) / M, np.sum(cappriceLS, axis=1) / M, se_Eul, se_LS

# tau       = Difference between tenor points
# V         = Instantaneous volatility matrix of the large time step method
# V2        = Instantaneous volatility matrix of the Euler method
# L         = Initial forward rates
# NoOfSteps = Number of steps used
# NoOfFor   = Number of forward rates that will be simulated
# T         = End date
# spotrate  = Spotrate
# M         = Number of Monte Carlo simulations
# EulSteps  = Number of Euler discretization steps
def zero_coupon_bond(tau, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M, EulSteps):
    """Calculate the ZCB rate analytically and using Monte Carlo simulation"""
    eurate    = np.zeros([NoOfFor, M])                                                      # Rate using Euler discretization
    lsrate    = np.zeros([NoOfFor, M])                                                      # Rate using a larger time step        
    analytZCB = np.zeros(NoOfFor)                                                        # Analytical rate   
    
    # Value to calculate the rate analytically 
    analytZCB[0] = (1 + tau * spotrate)
    for i in range(0,NoOfFor-1):
        analytZCB[i+1] = analytZCB[i] * (1 + tau * L[i])  
    
    # Perform Monte Carlo Simulation
    for i in range(M):
        FRW, FRWanti     = Generate_FRW_Eul(NoOfSteps*EulSteps, NoOfFor, T, tau, V2, L)        # Obtain forward rates using a Euler discretization
        FRWLS, FRWLSanti = Generate_FRW_LS(NoOfSteps, NoOfFor, T, tau, V, L)               # Obtain forward rates using a large time step
        for k in range(NoOfFor):                                                         # Loop to cover multiple ZCB       
            # Obtain the rate using Euler discretization
            pEul = pEulanti = (1 + tau * spotrate)                                       # First discount rate uses spotrate
            for j in range(0, k):
                pEul     = pEul * (1 + tau * FRW[j, (j+1)*EulSteps])
                pEulanti = pEulanti * (1 + tau * FRWanti[j, (j+1)*EulSteps])
            eurate[k, i] = (1 / pEul + 1 / pEulanti) * 0.5
            
            # Obtain the rate using large time step
            pLS = pLSanti = (1 + tau * spotrate)
            for j in range(0, k):
                pLS      = pLS * (1 + tau * FRWLS[j, j+1])
                pLSanti  = pLSanti * (1 + tau * FRWLSanti[j, j+1]) 
            lsrate[k, i] = (1 / pLS + 1 / pLSanti) * 0.5
    
    # Calculate standard errors
    se_Eul = np.sqrt(np.var(eurate, axis=1, ddof=1)) / np.sqrt(M)
    se_LS  = np.sqrt(np.var(lsrate, axis=1, ddof=1)) / np.sqrt(M)        
    
    # [Analytical rate, Rate using Euler, Rate using large steps]
    return 1 / analytZCB, np.sum(eurate, axis=1) / M, np.sum(lsrate,axis=1) / M, se_Eul, se_LS

def mainCalculation():
    """Run the program to generate forward rates."""    
    tau       = 0.25                                                                # Difference between time steps
    T         = 1                                                                   # Time horizon
    EulSteps  = 100                                                                # Euler discretization steps between two tenor points
    
    NoOfSteps = int(T / tau)                                                        # Number of time-steps                                                    
    NoOfFor = int(T / tau)                                                          # Number of forward rates we want to generate
    
    # Note that normally you will obtain these using calibration, now I simulate them randomly
    V = insVol(NoOfSteps, NoOfFor)
    V2 = insVol(NoOfSteps*EulSteps, NoOfFor)
    L = inFor(NoOfFor)

    FRW, FRWanti = Generate_FRW_Eul(NoOfSteps*EulSteps, NoOfFor, T, tau, V2, L)                    # Generate forward rates using Euler discretization

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
    
    FRWLS, FRWLSanti = Generate_FRW_LS(NoOfSteps, NoOfFor, T, tau, V, L)                         # Generate forward rates using a large time-step
    
    # Plot the different forward rates
    plt.figure()
    for i in range(len(FRWLS)):
            plt.plot(FRWLS[i,:])
    plt.title('Forward rates using big time-steps')
    plt.ylabel('Forward rate')
    plt.xlabel('Time (years)')
    plt.xticks(locsLS, labels, rotation='45')
    plt.show()
 
    
def priceCaplet():
    """Run the program to price caplets."""    
    tau       = 0.25                                                                # Difference between time steps
    T         = 2                                                                   # Time horizon
    M         = 1000                                                                 # Number of Monte Carlo simulations
    EulSteps  = 64                                                                   # Euler discretization steps between two tenor points

    NoOfSteps = int(T / tau)                                                        # Number of time-steps                                                    
    NoOfFor   = int(T / tau)                                                        # Number of forward rates we want to generate
    
    # Note that normally you will obtain these using calibration, now I simulate them randomly
    V         = insVol(NoOfSteps, NoOfFor)
    V2        = insVol(NoOfSteps*EulSteps, NoOfFor)
    L         = inFor(NoOfFor)
    spotrate  = L[0] / 2
    
    K         = L                                                               # Strike price caplet
    Notional  = 10000
    
    [capEul, capLS, se_Eul, se_LS] = cap_price_MC(Notional, tau, K, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M, EulSteps)         # Obtain the simulated caplet prices
    capBlack = cap_price_Black(Notional, tau, T, K, V, L, spotrate)                                                  # Obtain the analytical caplet price  
 
    colname = list()
    for i in range(NoOfFor):
        name = '{}{}'.format('Cap', '(T_' + str(i+1) + ',' + 'T_' + str(i+2) + ')')
        colname.append(name)
  
    table1 = zip(colname, capBlack, capEul, 100 * abs(capEul - capBlack) / capBlack, se_Eul)
    table2 = zip(colname, capBlack, capLS, 100 * abs(capLS - capBlack) /  capBlack, se_LS)
    header1 = ['Analytical Price', 'Euler price', 'Error (%)', 'Standard\n error']
    header2 = ['Analytical Price', 'Predictor-Corrector price', 'Error (%)', 'Standard\n error']  
    
    print(tabulate(table1, headers = header1, tablefmt="fancy_grid"), '\n')
    print(tabulate(table2, headers = header2, tablefmt="fancy_grid"))
    
    # # Generate the plots
    # M     = [500, 1000, 5000, 10000, 50000]
    # capLS = np.zeros([len(M), NoOfFor])
    # se_LS = np.zeros([len(M), NoOfFor])
    # rel   = np.zeros([len(M), NoOfFor])
    
    # labels = [] ; locs = np.zeros(NoOfFor+1) ; locsLS = np.zeros(NoOfFor+1)
    # for i in range(NoOfFor):
    #     labels.append(r'$T_{%s}$' %(i+1))
    #     locs[i] = i * EulSteps
    #     locsLS[i] = i

    # # Plot the different forward rates
    # plt.figure()    
    # for i in range(len(M)):
    #     [a, prices, b, se] = cap_price_MC(Notional, tau, K, V, V2, L, NoOfSteps, NoOfFor, T, spotrate, M[i], EulSteps)         # Obtain the simulated caplet prices
    #     capLS[i, :]       = prices
    #     se_LS[i, :]       = se
    #     rel[i, :]         = 100 * abs(prices - capBlack) / prices
    
    # for i in range(len(M)):
    #     plt.plot(capLS[i, :], label='%d'%M[i])

    # plt.plot(capBlack, label='Blacks price')
    # plt.title('Simulated caplet prices')
    # plt.ylabel('Caplet price (bps)')
    # plt.xlabel('Reset time')
    # plt.xticks(locsLS,labels, rotation='45')
    # plt.legend(title='Number of simulations')
    # plt.grid()
    # plt.show()
    
    # plt.figure()
    # for i in range(len(M)):
    #     plt.plot(se_LS[i, :], label='%d'%M[i])

    # plt.title('Standard errors')
    # plt.ylabel('Standard error')
    # plt.xlabel('Reset time')
    # plt.xticks(locsLS,labels, rotation='45')
    # plt.legend(title='Number of simulations')
    # plt.grid()
    # plt.show()
    
    # plt.figure()
    # for i in range(len(M)):
    #     plt.plot(rel[i, :], label='%d'%M[i])

    # plt.title('Relative errors')
    # plt.ylabel('Relative error (%)')
    # plt.xlabel('Reset time')
    # plt.xticks(locsLS,labels, rotation='45')
    # plt.legend(title='Number of simulations')
    # plt.grid()
    # plt.show()
    
    
def priceZCB(): 
    """Run the program to generate discount rates."""    
    tau       = 0.25                                                                # Difference between time steps in years
    T         = 2                                                                   # Time horizon in years
    M         = 500                                                                 # Number of Monte Carlo simulations
    EulSteps  = 100                                                                 # Euler discretization steps

    NoOfSteps = int(T / tau)                                                        # Number of time-steps                                                    
    NoOfFor   = int(T / tau)                                                        # Number of forward rates we want to generate
    
    # Note that normally you will obtain these using calibration, now I simulate them randomly
    V         = insVol(NoOfSteps, NoOfFor)
    V2        = insVol(NoOfSteps*EulSteps, NoOfFor)
    L         = inFor(NoOfFor)
    spotrate  = L[0] / 2 
      
    [anrate, eurate, lsrate, se_Eul, se_LS] = zero_coupon_bond(tau, V, V2, L, 
                                                NoOfSteps, NoOfFor, T, spotrate, M, EulSteps)      # Obtain the differenct calculatd zero-coupon rates
    
    colname = list()
    for i in range(NoOfFor):
        name = '{}{}'.format('P', '(T_' + str(i) + ')')
        colname.append(name)
  
    table1 = zip(colname, anrate, eurate, 100 * abs(eurate - anrate) / anrate, se_Eul)
    table2 = zip(colname, anrate, lsrate, 100 * abs(lsrate - anrate) / anrate, se_LS)
    header1 = ['Analytical Price', 'Euler price', 'Eul error (%)', 'Standard\n error']
    header2 = ['Analytical Price', 'Log-Euler price', 'Log-Eul error (%)', 'Standard\n error']  
    
    print(tabulate(table1, headers = header1, tablefmt="fancy_grid"), '\n')
    print(tabulate(table2, headers = header2, tablefmt="fancy_grid"))
    
    
    # Generate the plots
    M     = [500, 1000, 5000, 10000, 50000]
    capLS = np.zeros([len(M), NoOfFor])
    se_LS = np.zeros([len(M), NoOfFor])
    rel   = np.zeros([len(M), NoOfFor])
    
    labels = [] ; locs = np.zeros(NoOfFor+1) ; locsLS = np.zeros(NoOfFor+1)
    for i in range(NoOfFor):
        labels.append(r'$T_{%s}$' %(i+1))
        locs[i] = i * EulSteps
        locsLS[i] = i

    # # Plot the ZCB validation
    # plt.figure()    
    # for i in range(len(M)):
    #     [anrate, eurate, lsrate, se_Eul, se] = zero_coupon_bond(tau, V, V2, L, 
    #                                             NoOfSteps, NoOfFor, T, spotrate, M[i], EulSteps)      # Obtain the differenct calculatd zero-coupon rates

    #     capLS[i, :]       = lsrate
    #     se_LS[i, :]       = se
    #     rel[i, :]         = 100 * abs(lsrate - anrate) / lsrate
    
    # for i in range(len(M)):
    #     plt.plot(capLS[i, :], label='%d'%M[i])

    # plt.plot(anrate, label='Market price')
    # plt.title('Simulated ZCB prices')
    # plt.ylabel('ZCB value')
    # plt.xlabel('Maturity time')
    # plt.xticks(locsLS,labels, rotation='45')
    # plt.legend(title='Number of simulations')
    # plt.grid()
    # plt.show()
    
    # plt.figure()
    # for i in range(len(M)):
    #     plt.plot(se_LS[i, :], label='%d'%M[i])

    # plt.title('Standard errors')
    # plt.ylabel('Standard error')
    # plt.xlabel('Maturity time')
    # plt.xticks(locsLS,labels, rotation='45')
    # plt.legend(title='Number of simulations')
    # plt.grid()
    # plt.show()
    
    # plt.figure()
    # for i in range(len(M)):
    #     plt.plot(rel[i, :], label='%d'%M[i])

    # plt.title('Relative errors')
    # plt.ylabel('Relative error (%)')
    # plt.xlabel('Maturity time')
    # plt.xticks(locsLS,labels, rotation='45')
    # plt.legend(title='Number of simulations')
    # plt.grid()
    # plt.show()

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
priceCaplet()
stop = timeit.default_timer() 
print('\n Time: ', stop - start, '\n')

#%% Run the discount rate function
np.random.seed(seed)                                                                  # Set seed

start = timeit.default_timer()
print('Starting the zero-coupon bond pricing function \n')    
priceZCB()
stop = timeit.default_timer() 
print('\n Time: ', stop - start)