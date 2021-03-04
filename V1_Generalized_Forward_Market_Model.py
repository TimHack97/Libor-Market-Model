import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm
from random import randint
from tabulate import tabulate

seed = 1
# seed = randint(0, 100000)                                                           # Create random seed

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
def insVol(NoOfSteps, NoOfRates): 
    """Generate the instantaneous volatility matrix."""
    np.random.seed(seed)
    Vinit = np.random.uniform(0.2, 0.4, NoOfRates)                                  # Draw from a uniform distribution every IV for every forward rate
    V = np.zeros([NoOfRates, NoOfSteps]) 
       
    tenor_steps = NoOfSteps / NoOfRates                                             # Steps in between tenor points
    
    for i in range(NoOfSteps):                                                      # Loop per time step
        for j in range(NoOfRates):                                                  # Loop per forward rate
            if i == 0:
                V[j,:] = Vinit[j]                                                   # Set every time step equal to the initialized IV
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

# NoOfRates = Number of rates
def inFor(NoOfRates): 
    """Generate the initial forward rates."""        
    L = np.random.uniform(0.02, 0.08, NoOfRates)                                    # Draw from an uniform distribution between [0.2,0.8]
    L = np.sort(L)                                                                  # Sort the values such that the shortest rate has the lowest value

    return L                                                                        # Return the forward rates

# V         = Instantaneous volatility matrix
# IC        = Instantaneous correlation matrix
# time      = Current time
# rate      = Backward rate
# sumrate   = Rate showing in the sum
# tau       = Size time-step
def getvalueC(V, IC, time, rate, sumrate, tau):   
    """Approximate the integral over the time-step from the instantaneous volatilties and correlation"""
    
    # Note that I can just use the first value of the matrices since the NaN 
    # is taken care of in the other function and the IV is constant
    fa = V[rate, 0] * V[sumrate, 0] * \
        IC[rate, sumrate]                                                        # Left part of the integral
    fb = V[rate, 0] * V[sumrate, 0] * \
        IC[rate, sumrate]                                                        # Right part of the integral
    
    Cval = tau * ((fa + fb) / 2)                                                 # Trapezoidal approximation of an integral

    return Cval                                                                  # Return the approximated integral

# tau           = Difference between time steps
# time          = Current time
# BWR           = Matrix with the till 'time' calculated backward rates
# bottom        = First value of the summation
# rate          = Number of the current backward rate
# V             = Instantaneous volatility matrix
# IC            = Instantaneous correlation matrix
# timeIntegral  = Time for the integral Cij, this is different from the time for v(T_k) when using the better drift approx.
def approxDrift(tau, BWR, bottom, rate, V, IC, time, timeIntegral):
    """Approximate the drift integral"""
    X = 0                                                                       # Approximation value
    for k in range(bottom, rate+1):
        sumrate = k                                                             # Current value of the summation of the drift
        Cij = getvalueC(V, IC, timeIntegral, rate, sumrate, tau)             # Calculate the C integral
        if time == timeIntegral:                                                # This means setting R(u) = R(k)
            X += (((tau * BWR[k,time]) /    \
                 (1 + tau * BWR[k,time])) * Cij)                                # Approximate the drift formula
        else:                                                                   # Predictor-Corrector approximation
            approxFRW = 0.5 * (BWR[k,time] + BWR[k,time-1])
            X += (((tau * approxFRW) /    \
                 (1 + tau * approxFRW)) * Cij)                                  # Approximate the drift formula            
    return X

# NoOfSteps     = Number of time steps
# NoOfRat       = Number of rates
# T             = End time
# tau           = Difference between time steps
# V             = Instantaneous volatility matrix
# L0            = Initial forward rates
def Generate_BWR_Log_Eul(NoOfSteps, NoOfRat, T, tau, V, L0):
    """Generate backward rates using the log-Euler method."""
    # Obtain the initial values needed for the simulation
    IC = insCorr(NoOfRat)                                                            # Instantaneous correlation matrix
    BM, BManti = GenerateBM(T, NoOfSteps, NoOfRat, IC)                               # Generated Brownian motions
    
    # Initialize the forward rates
    BWR = np.zeros([NoOfRat, NoOfSteps+1]) 
    BWR[:,0] = np.transpose(L0) 
    
    BWRanti = np.zeros([NoOfRat, NoOfSteps+1]) 
    BWRanti[:,0] = np.transpose(L0)     
        
    for time in range(NoOfSteps):
        for rate in range(NoOfRat): 
            q = np.int(np.ceil((time*NoOfRat) / NoOfSteps + 0.01))                  # Smallest tenor point that is larger than the current time step
            Cii = getvalueC(V, IC, time, rate, rate, tau)
            if q <= (rate+1):                                                       # We say here rate+1 since in the literature the first rate is always 1 but python starts at 0
                                                                                    # Since the summation q starts at T_1 = 1 we have to start at rate 1
                X1 = approxDrift(tau, BWR, (q-1), rate, V, IC, time, time)          # q-1 is the bottom of the summation sign in drift
                Y = -0.5 * Cii
                Z = V[rate, 0] * (BM[rate , time+1] - BM[rate , time])              # Note: I just use the first volatility value since it is constant over time

                BWR[rate,time+1] = BWR[rate,time] * np.exp(X1 + Y + Z)  

                # Perform better approximation (Predictor-Corrector)
                X2 = approxDrift(tau, BWR, (q-1), rate, V, IC, time+1, time)
                BWR[rate,time+1] = BWR[rate,time] * np.exp(X2 + Y + Z)
                
                # Now using antithetic variables
                X1anti = approxDrift(tau, BWRanti, (q-1), rate, V, IC, time, time)             # q-1 is the bottom of the summation sign
                Yanti = -0.5 * Cii
                Zanti = V[rate, 0] * (BManti[rate , time+1] - BManti[rate , time])

                BWRanti[rate,time+1] = BWRanti[rate,time] * np.exp(X1anti + Yanti + Zanti)  

                # Perform better approximation (Predictor-Corrector)
                X2anti = approxDrift(tau, BWRanti, (q-1), rate, V, IC, time+1, time)
                BWRanti[rate,time+1] = BWRanti[rate,time] * np.exp(X2anti + Yanti + Zanti)
            else:
                BWR[rate,time+1] = np.nan
                BWRanti[rate,time+1] = np.nan
    
    return BWR, BWRanti, IC                                                                        # Return backward rates

# NoOfSteps     = Number of time steps
# NoOfRat       = Number of rates
# T             = End time
# tau           = Difference between time steps
# V             = Instantaneous volatility matrix
# L0            = Initial forward rates
# eul_steps     = Number of euler discretization steps
def Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, V, L0, eul_steps):
    """Generate backward-looking forward rates"""
    BWR, BWRanti, IC = Generate_BWR_Log_Eul(NoOfSteps, NoOfRates, T, tau, V, L0)                   # Obtain backward rates which were generated using the log_euler method
    spotrate = L0[0] / 2                                                                           # Self-picked spotrate

    # Initialize the backward rates
    BW_rate = np.zeros([NoOfRates, NoOfSteps + eul_steps])                             
    BW_rate[:,0] = np.transpose(L0) 
    
    BW_rate_anti = np.zeros([NoOfRates, NoOfSteps + eul_steps]) 
    BW_rate_anti[:,0] = np.transpose(L0) 
    
    # Every rate needs it's own 'x-axis'
    x_axis = np.zeros([NoOfRates, NoOfSteps + eul_steps])
       
    # Set values up to time T_(j-1) to the same value as forward rates
    """For t <= T_j the backward rates are equal to the log-Euler rates this
    is the same as the generated forward rates for the LMM."""
    for i in range(NoOfRates):
        if i == 0:
            BW_rate[i, 0:i+1] = spotrate
            BW_rate_anti[i, 0:i+1] = spotrate
            x_axis[i, 0:i+1]   = 0
        else: 
            BW_rate[i, 0:i+1] = BWR[i-1, 0:i+1]
            BW_rate_anti[i, 0:i+1] = BWRanti[i-1, 0:i+1]
            x_axis[i, 0:i+1]   = np.arange(0, i+1) * tau

    # Euler discretization size
    dt = tau / eul_steps                                                                # time step
        
    BM_new, BManti_new = GenerateBM(T, eul_steps, NoOfRates, IC)                        # Create new Brownian motions
    g = lambda Tj, t: min( (max(Tj-t,0)) / (Tj - (Tj-tau) ) , 1 )                       # Function for decreasing volatility  
    
  
    for steps in range(eul_steps):                                                      # Loop per Euler step
        for rate in range(NoOfRates):                                                   # Loop per forward rate
            """The steps+1+rate means that we start at the next point after T_(j-1)"""
            """This point is ofcourse different per rate, that is why we have the +rate"""
            """The steps are the Euler discretization steps"""
            x_axis[rate, steps+1+rate] = x_axis[rate, steps+rate] + dt                    # Add the small time step to the x-axis
           
            Tj    = x_axis[rate, rate] + tau                                             # T_j
            t     = x_axis[rate, steps+rate]                                             # Current time
            
            gamma_rate = g(Tj, t)   
            X     = Xanti = 0
            q     = int(np.ceil(((Tj * (1 / tau)) + (dt / 10))))                        # q according to Piterbarg's notation
            
            # We actually never use this drift equation since in the accrual period the drift is zero. I will probably remove it
            for j in range(q-1, rate+1):                                                # Perform the summation of the drift term
                """The summation is from q(t) to j. Note that this depends on your tenor structure. I assume Piterbargs structure"""
                """So this is different from the paper. By the definition of q(t) it makes sense. The -1 since python starts at zero."""
                Tj_new = x_axis[j, rate] + tau                                          # Tenor point for the summation
                gamma = g(Tj_new, t)                                             
                
                "I always use V[j,0], since the instantaneous volatility is constant I can just take the first value in column 0."
                Y = tau * IC[rate,j] * V[j,0] * BW_rate[j,steps+rate] * gamma
                Z = 1 + tau * BW_rate[j,steps+rate]
                X += Y / Z

                Yanti = tau * IC[rate,j] * V[j,0] * BW_rate_anti[j,steps+rate] * gamma
                Zanti = 1 + tau * BW_rate_anti[j,steps+rate]
                Xanti += Yanti / Zanti    
            
            BW_rate[rate,steps+1+rate] = BW_rate[rate,steps+rate] + V[rate,0] * gamma_rate * BW_rate[rate,steps+rate] * X * dt + \
                V[rate,0] * BW_rate[rate,steps+rate] * gamma_rate * (BM_new[rate,steps+1] - BM_new[rate,steps])                          # Perform the Euler discretization
            
            BW_rate_anti[rate,steps+1+rate] = BW_rate_anti[rate,steps+rate] + V[rate,0] * gamma_rate * BW_rate_anti[rate,steps+rate] * Xanti * dt + \
                V[rate,0] * BW_rate_anti[rate,steps+rate] * gamma_rate * (BManti_new[rate,steps+1] - BManti_new[rate,steps])                          # Perform the Euler discretization
                           
    # Define values after the accrual period
    for rates in range(NoOfRates):                                                                      # Loop per backward rate
        for step in range(NoOfRates + eul_steps - (NoOfRates - rates - 1), NoOfRates + eul_steps):     # Loop that starts at time R(T_j, T_{j-1}, T_j) and continues till end point
            x_axis[rates, step]   = x_axis[rates, step-1] + tau                                         # Extend x-axis after accrual period with steps of tau
            
            # Set forward-looking backward rates after accrual period
            # BW_rate[rates, step] = BW_rate[rates, step-1] 
            BW_rate[rates, step] = np.nan
            
            # BW_rate_anti[rates, step] = BW_rate[rates, step-1] 
            BW_rate_anti[rates, step] = np.nan


    return BW_rate, BW_rate_anti, x_axis                                                                 # Return the forward-looking backward rates and their corresponding x-axis

# Notional  = Notional of the contract
# tau       = Difference between tenor points
# T         = Reset date of the caplet
# K         = Strike price of the caplet
# V         = Instantaneous volatility matrix
# L         = Initional forward rates
def cap_price_Black(Notional, tau, T, K, V, L):  
    """Calculate the price of a caplet using Black's formula"""
    spotrate = L[0] / 2
    
    num_of_caps = int(T/tau)                                                                    # Number of caplets we want to validate
    cap_price = np.zeros(num_of_caps)                                                           # Save caplet prices
    
    for resetT in range(0, num_of_caps):                                                        # Apply Black's formula for every caplet                   
        helpval = (tau ** 3) / (tau ** 2)                                                       # We want the caplet price at time zero hence no maximum needed
    
        vsqr = (resetT * tau + 1 / 3 * helpval) * V[resetT, 0] * V[resetT, 0]                   # ResetT * tau since ResetT = 0, 1, 2 etc.. but in terms of T_0, T_1, T_2 = 0, 0.25, 0.5
        
        if resetT == 0:
            R = spotrate
        else: 
            R = L[resetT-1]
            
        d1 = (np.log(R / K) + (vsqr / 2)) / (np.sqrt(vsqr)) 
        d2 = d1 - (np.sqrt(vsqr))
        
        P = (1 / (1 + tau * spotrate))
        for i in range(0,resetT):      
            # Loop to calculate the value of P(0,T_n)
            P = P * (1 / (1 + tau * L[i]) )
               
        cap_price[resetT] = Notional * P * (R * norm.cdf(d1) - K * norm.cdf(d2) )

    return cap_price

# N         = Notional
# tau       = Difference between tenor points
# resetT    = Reset date of the caplet
# K         = Strike price of the caplet
# V         = Instantaneous volatility matrix of the large time step method
# L         = Initial forward rates
# NoOfSteps = Number of steps used
# NoOfRates = Number of rates that will be simulated
# T         = End date
# M         = Number of Monte Carlo simulations
# eul_steps = Number of Euler discretization steps
def cap_price_MC(N, tau, K, V, L, NoOfSteps, NoOfRates, T, M, eul_steps):
    """Calculate the price of a caplet using Monte Carlo simulation"""   
    cappriceBack = np.zeros([NoOfRates, M])                                                  # Save caplet prices for backward-looking caplet
    
    for i in range(M):
        [BW_rate, BW_rate_anti, xaxis] = Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, V, L, eul_steps)                 # Obtain backward rates using a large time step
        for resetT in range(0, NoOfRates):                                                   # Calculate every caplet price
            # Calculate the payoffs using the forward rates from the large step method
            payoffLS     = N * max(BW_rate[resetT, resetT + eul_steps] - K, 0)
            payoffLSanti = N * max(BW_rate_anti[resetT, resetT + eul_steps] - K, 0)
            
            discountLS = discountLSanti = 1
            for k in range(0,resetT+1):
                discountLS     = discountLS * (1 + tau * BW_rate[k,k + eul_steps])
                discountLSanti = discountLSanti * (1 + tau * BW_rate_anti[k,k + eul_steps])

            cappriceBack[resetT, i] = (payoffLS / discountLS + payoffLSanti / discountLSanti) * 0.5
    
    # Calculate standard errors
    se_Back  = np.sqrt(np.var(cappriceBack, axis=1, ddof=1)) / np.sqrt(M)       
    
    return np.sum(cappriceBack, axis=1) / M, se_Back

# tau       = Difference between tenor points
# V         = Instantaneous volatility matrix of the large time step method
# L         = Initial forward rates
# NoOfSteps = Number of steps used
# NoOfRat   = Number of rates that will be simulated
# T         = End date
# M         = Number of Monte Carlo simulations
# eul_steps  = Number of Euler discretization steps
def zero_coupon_bond(tau, V, L, NoOfSteps, NoOfRat, T, M, eul_steps):
    """Calculate the ZCB rate analytically and using Monte Carlo simulation"""
    discount_rate = np.zeros([NoOfRat, M])                                                    # Simulated ZCB
    analytZCB = np.zeros(NoOfRat)                                                        # Analytical ZCB   
    
    # Value to calculate the rate analytically 
    analytZCB[0] = (1 + tau * (L[0] / 2))                                                # L[0] / 2 = spotrat  
    for i in range(0,NoOfRat-1):
        analytZCB[i+1] = analytZCB[i] * (1 + tau * L[i])  
    
    # Perform Monte Carlo Simulation
    for i in range(M):
        [BW_rate, BW_rate_anti, x_axis] = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps)       # Obtain backward rates using simulation
        for k in range(NoOfRat):                                                         # Loop to cover multiple ZCB       
            # Obtain the rate using Euler discretization
            pSim = pSimanti = 1                                     # First discount rate uses spotrate
            for j in range(0, k+1):
                pSim = pSim * (1 + tau * BW_rate[j, j + eul_steps])
                pSimanti = pSimanti * (1 + tau * BW_rate_anti[j, j + eul_steps])
            discount_rate[k, i] = (1 / pSim + 1 / pSimanti) * 0.5
            
    
    # Calculate standard errors
    se_Sim = np.sqrt(np.var(discount_rate, axis=1, ddof=1)) / np.sqrt(M)
    
    # [Analytical rate, Rate using Euler, Rate using large steps]
    return 1 / analytZCB, np.sum(discount_rate, axis=1) / M, se_Sim


def mainCalculation():
    """Run the program to generate backward rates."""    
    tau         = 0.25                                                              # Difference between tenor points
    T           = 1                                                                 # Time horizon
    eul_steps   = 50                                                                # Euler discretization steps between two tenor points
    
    NoOfSteps = int(T / tau)                                                        # Number of time-steps                                                    
    NoOfRat = int(T / tau)                                                          # Number of backward rates we want to generate

    # Note that normally you will obtain these using calibration, now I simulate them randomly
    V = insVol(NoOfSteps, NoOfRat)                                                  # Instantaneous volatility
    L = inFor(NoOfRat)                                                              # Initial rates

    # Obtain the forward-looking backward rates and their corresponding x-axis
    BW_rate, BW_rate_anti, x_axis = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps)
    
    "Plotting the rates"
    # Obtain labels and locations for the x-axis
    labels = np.zeros(NoOfRat+1) ; locs = np.zeros(NoOfRat+1) 
    hv = 0
    for i in range(NoOfRat+1):
        labels[i] = hv
        locs[i] = i * tau
        hv += 0.25
      
    # Plot the backward-rates
    fig, ax = plt.subplots()
    for i in range(NoOfRat):
        ax.plot(x_axis[i,:], BW_rate[i,:])
        
    plt.title('Simulated backward-looking forward rates')
    plt.ylabel('Rate (%)')
    plt.xlabel('Time (years)')
    plt.xticks(locs,labels, rotation='45')


    "Validating caplet prices"
    Notional = 1000                                                                 # Notional
    K = 0.01                                                                        # Strike price
    M = 500                                                                         # Number of Monte Carlo simulations

    [capLS, se_LS] = cap_price_MC(Notional, tau, K, V, L, NoOfSteps, NoOfRat, T, M, eul_steps)         # Obtain the simulated caplet prices
    capBlack = cap_price_Black(Notional, tau, T, K, V, L)   
    
    # Generate nice ouput
    colname = list()
    for i in range(NoOfRat):
        name = '{}{}'.format('Cap', '(T_' + str(i) + ',' + 'T_' + str(i+1) + ')')
        colname.append(name)
  
    table = zip(colname, capBlack, capLS, 100 * abs(capLS - capBlack) /  capBlack, se_LS)
    header = ['Analytical Price', 'Simulated price', 'Error (%)', 'Standard\n error']  
    
    print(tabulate(table, headers = header, tablefmt="fancy_grid"))
    
    
    "Valiating zero-coupon bond prices"
    M = 500
    
    [anrate, sim_rate, se_Sim] = zero_coupon_bond(tau, V, L, NoOfSteps, NoOfRat, T, M, eul_steps)      # Obtain the differenct calculatd zero-coupon rates
    
    colname = list()
    for i in range(NoOfRat):
        name = '{}{}'.format('P', '(T_' + str(i+1) + ')')
        colname.append(name)
  
    table = zip(colname, anrate, sim_rate, 100 * abs(sim_rate - anrate) / anrate, se_Sim)
    header = ['Analytical Price', 'Simulated price', 'Error (%)', 'Standard error']
    
    print(tabulate(table, headers = header, tablefmt="fancy_grid"))
    
    
# Time the program           
start = timeit.default_timer()   
print('Starting the function') 
mainCalculation()
stop = timeit.default_timer() 
print('Time: ', stop - start, '\n')