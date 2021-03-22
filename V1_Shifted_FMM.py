import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm
from random import randint
from tabulate import tabulate
import seaborn as sns
from scipy import stats
import scipy.optimize as optimize


# seed = 1
seed = randint(0, 100000)                                                         # Create random seed

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

    Vol = 0.3                                                                       # Specify the volatility for the backward rate running from [T_0, T_1]

    return V, Vol                                                                   # Return the instantaneous volatility matrix

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
# Rate_type = Either positive or negative starting rates
def inFor(NoOfRates, rate_type): 
    """Generate the initial forward rates."""      
    if rate_type == 'positive':                                                     # Generate positive rates
        L = np.random.uniform(0.02, 0.08, NoOfRates)                                # Draw from an uniform distribution between [0.2,0.8]
        L = np.sort(L)                                                              # Sort the values such that the shortest rate has the lowest value
    else:                                                                           # Generate negative rates
        L = np.random.uniform(-0.005, -0.03, NoOfRates)
        L = np.sort(L)
   
    return L                                                                        # Return the forward rates
# NoOfRates = Number of rates
# L         = Initial rates
# spotrate  = spotrate
def theta(NoOfRates, L, spotrate):  
    """Generate the shift needed to make the starting rates positive"""
    theta_logE = np.multiply(L, -1) + np.linspace(0.02, 0.08, NoOfRates)                    # Shifts used for the log-Euler step. 
    
    """Note that we used the same shift for the methods however for the Euler discretization we
    have to take into consideration that the first backward rate comes from the spotrate while
    for the log-Euler method we basically simulated forward rates so we don't work with the spotrate yet."""
    theta_Eul = np.roll([np.multiply(L, -1) + np.linspace(0.02, 0.08, NoOfRates)],1)        # Shifts used for Euler discretization
    theta_Eul[0,0] = spotrate * -1 + 0.01
    
    return theta_logE, theta_Eul

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
# BWR_org       = Matrix with the till 'time' calculated original (negative) backward rates = forward rates for the entirty of log-Euler
# bottom        = First value of the summation
# rate          = Number of the current backward rate
# V             = Instantaneous volatility matrix
# IC            = Instantaneous correlation matrix
# timeIntegral  = Time for the integral Cij, this is different from the time for v(T_k) when using the better drift approx.
# BWR_shift     = Matrix with the till 'time' calculated shifted backward rates = forward rates for the entirty of log-Euler
# theta         = Shift values
def approxDrift(tau, BWR_org, bottom, rate, V, IC, time, timeIntegral, BWR_shift, theta):
    """Approximate the drift integral"""
    X = 0                                                                       # Approximation value
    for k in range(bottom, rate+1):
        sumrate = k                                                             # Current value of the summation of the drift
        Cij = getvalueC(V, IC, timeIntegral, rate, sumrate, tau)                # Calculate the C integral
        if time == timeIntegral:                                                # This means setting R(u) = R(k)
            X += (((tau * (BWR_org[k,time]+theta[k])) /    \
                 (1 + tau * BWR_org[k,time])) * Cij)                                # Approximate the drift formula
        else:                                                                   # Predictor-Corrector approximation
            approxFRW = 0.5 * ((BWR_org[k,time] + theta[k]) + (BWR_org[k,time-1] + theta[k]))
            approxFRW_org = 0.5 * ((BWR_org[k,time]) + (BWR_org[k,time-1]))
            X += (((tau * approxFRW) /    \
                 (1 + tau * approxFRW_org)) * Cij)                                  # Approximate the drift formula            
    return X

# NoOfSteps     = Number of time steps
# NoOfRat       = Number of rates
# T             = End time
# tau           = Difference between time steps
# V             = Instantaneous volatility matrix
# L0            = Initial forward rates
# theta         = Shift values
def Generate_BWR_Log_Eul(NoOfSteps, NoOfRat, T, tau, V, L0, theta):
    """Generate backward rates using the log-Euler method."""
    """You can actually see this as generating forward rates then in the next function we put the right values at the correct places"""
    # Obtain the initial values needed for the simulation
    IC = insCorr(NoOfRat)                                                            # Instantaneous correlation matrix
    BM, BManti = GenerateBM(T, NoOfSteps, NoOfRat, IC)                               # Generated Brownian motions
    
    L0_original = L0                                                                 # Initial negative rates
    L0_shifted  = L0 + theta                                                         # Shifted positive rates
    
    
    # Initialize the forward rates
    BWR_original = np.zeros([NoOfRat, NoOfSteps+1]) 
    BWR_original[:,0] = np.transpose(L0_original) 
    
    BWRanti_original = np.zeros([NoOfRat, NoOfSteps+1]) 
    BWRanti_original[:,0] = np.transpose(L0_original)  
    
    BWR_shifted = np.zeros([NoOfRat, NoOfSteps+1]) 
    BWR_shifted[:,0] = np.transpose(L0_shifted) 
    
    BWRanti_shifted = np.zeros([NoOfRat, NoOfSteps+1]) 
    BWRanti_shifted[:,0] = np.transpose(L0_shifted)   
    
    
        
    for time in range(NoOfSteps):
        for rate in range(NoOfRat): 
            q = np.int(np.ceil((time*NoOfRat) / NoOfSteps + 0.01))                  # Smallest tenor point that is larger than the current time step
            Cii = getvalueC(V, IC, time, rate, rate, tau)
            if q <= (rate+1):                                                       # We say here rate+1 since in the literature the first rate is always 1 but python starts at 0
                                                                                    # Since the summation q starts at T_1 = 1 we have to start at rate 1
                X1 = approxDrift(tau, BWR_original, (q-1), rate, V, IC, time, time, BWR_shifted, theta)          # q-1 is the bottom of the summation sign in drift
                Y = -0.5 * Cii
                Z = V[rate, 0] * (BM[rate , time+1] - BM[rate , time])              # Note: I just use the first volatility value since it is constant over time

                BWR_shifted[rate,time+1] = BWR_shifted[rate,time] * np.exp(X1 + Y + Z)  # Calculate the next step of the shifted lognormal rates
                BWR_original[rate,time+1] = BWR_shifted[rate,time+1] - theta[rate]      # Shift back to obtain the original rates, needed for the dynamics of the shifted rates

                # Perform better approximation (Predictor-Corrector)
                X2 = approxDrift(tau, BWR_original, (q-1), rate, V, IC, time+1, time, BWR_shifted, theta)
                BWR_shifted[rate,time+1] = BWR_shifted[rate,time] * np.exp(X2 + Y + Z)
                BWR_original[rate,time+1] = BWR_shifted[rate,time+1] - theta[rate]
                
                # Now using antithetic variables
                X1anti = approxDrift(tau, BWRanti_original, (q-1), rate, V, IC, time, time, BWRanti_shifted, theta)             # q-1 is the bottom of the summation sign
                Yanti = -0.5 * Cii
                Zanti = V[rate, 0] * (BManti[rate , time+1] - BManti[rate , time])

                BWRanti_shifted[rate,time+1] = BWRanti_shifted[rate,time] * np.exp(X1anti + Yanti + Zanti)  
                BWRanti_original[rate,time+1] = BWRanti_shifted[rate,time+1] - theta[rate]

                # Perform better approximation (Predictor-Corrector)
                X2anti = approxDrift(tau, BWRanti_original, (q-1), rate, V, IC, time+1, time, BWRanti_shifted, theta)
                BWRanti_shifted[rate,time+1] = BWRanti_shifted[rate,time] * np.exp(X2anti + Yanti + Zanti)
                BWRanti_original[rate,time+1] = BWRanti_shifted[rate,time+1] - theta[rate]
            else:
                BWR_original[rate,time+1] = BWR_shifted[rate,time+1] = np.nan
                BWRanti_original[rate,time+1] = BWRanti_shifted[rate,time+1] = np.nan
       
    
    return BWR_original, BWRanti_original, BWR_shifted, BWRanti_shifted, IC                                                                        # Return backward rates

# NoOfSteps     = Number of time steps
# NoOfRat       = Number of rates
# T             = End time
# tau           = Difference between time steps
# V             = Instantaneous volatility matrix
# L0            = Initial forward rates
# eul_steps     = Number of euler discretization steps
# vol           = Volatility of the first backward rate
# Spotrate      = Spotrate
# theta_logE    = Shift used however no shift for spotrate taken into account
# theta_Eul     = Shift which contains a shift for the spotrate
def Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, V, L0, eul_steps, vol, spotrate, theta_logE, theta_Eul):
    """Generate backward-looking forward rates"""
    BWR_original, BWRanti_original, BWR_shifted, BWRanti_shifted, IC  = Generate_BWR_Log_Eul(NoOfSteps, NoOfRates, T, tau, V, L0, theta_logE)                   # Obtain backward rates which were generated using the log_euler method

    # Initialize the backward rates
    BW_rate_original      = np.zeros([NoOfRates, NoOfSteps + eul_steps])                                 
    BW_rate_anti_original = np.zeros([NoOfRates, NoOfSteps + eul_steps]) 

    BW_rate_shifted      = np.zeros([NoOfRates, NoOfSteps + eul_steps])                                 
    BW_rate_anti_shifted = np.zeros([NoOfRates, NoOfSteps + eul_steps]) 

    # Every rate needs it's own 'x-axis'
    x_axis = np.zeros([NoOfRates, NoOfSteps + eul_steps])
       
    # Set values up to time T_(j-1) to the same value as forward rates
    """For t <= T_j the backward rates are equal to the log-Euler rates this
    is the same as the generated forward rates for the LMM."""
    for i in range(NoOfRates):
        if i == 0: # Special case 
            BW_rate_original[i, 0:i+1] = BW_rate_anti_original[i, 0:i+1] = spotrate
            BW_rate_shifted[i, 0:i+1]  = BW_rate_anti_shifted[i,0:i+1]   = spotrate + theta_Eul[0,0]
            x_axis[i, 0:i+1]           = 0
        else: 
            BW_rate_original[i, 0:i+1]      = BWR_original[i-1, 0:i+1]
            BW_rate_anti_original[i, 0:i+1] = BWRanti_original[i-1, 0:i+1]
            
            BW_rate_shifted[i, 0:i+1]      = BWR_shifted[i-1, 0:i+1]
            BW_rate_anti_shifted[i, 0:i+1] = BWRanti_shifted[i-1, 0:i+1]
            x_axis[i, 0:i+1]               = np.arange(0, i+1) * tau

    # Euler discretization size
    dt = tau / eul_steps                                                                # time step
        
    BM_new, BManti_new = GenerateBM(tau, eul_steps, NoOfRates, IC)                      # Create new Brownian motions
    g = lambda Tj, t: min( (max(Tj-t,0)) / (Tj - (Tj-tau) ) , 1 )                       # Function for decreasing volatility  
    
  
    for steps in range(eul_steps):                                                      # Loop per Euler step
        for rate in range(NoOfRates):                                                   # Loop per forward rate
            """The steps+1+rate means that we start at the next point after T_(j-1)"""
            """This point is ofcourse different per rate, that is why we have the +rate"""
            """The steps are the Euler discretization steps"""
            
            if rate > 0:
            
                x_axis[rate, steps+1+rate] = x_axis[rate, steps+rate] + dt                   # Add the small time step to the x-axis
               
                Tj    = x_axis[rate, rate] + tau                                             # T_j
                t     = x_axis[rate, steps+rate]                                             # Current time
                
                gamma_rate = g(Tj, t)   
                X     = Xanti = 0
                q     = int(np.ceil(((Tj * (1 / tau)) + (dt / 10))))                         # q according to Piterbarg's notation
                
                if steps == 0:                                                               # First step T_{j-1} we want the drift to come in
                    q = q-1
    
                for j in range(q-1, rate+1):                                                 # Perform the summation of the drift term
                    """The summation is from q(t) to j. Note that this depends on your tenor structure. I assume Piterbargs structure"""
                    """So this is different from the paper. By the definition of q(t) it makes sense. The -1 since python starts at zero."""                
                    Tj_new = x_axis[j, rate] + tau                                           # Tenor point for the summation
                    gamma = g(Tj_new, t)                                             
 
                    "I always use V[j,0], since the instantaneous volatility is constant I can just take the first value in column 0."
                    Y = tau * IC[rate-1,j-1] * V[j-1,0] * (BW_rate_original[j,steps+rate] + theta_Eul[0,j]) * gamma
                    Z = 1 + tau * BW_rate_original[j,steps+rate]
                    X += Y / Z
    
                    Yanti = tau * IC[rate-1,j-1] * V[j-1,0] * (BW_rate_anti_original[j,steps+rate] + theta_Eul[0,j]) * gamma
                    Zanti = 1 + tau * BW_rate_anti_original[j,steps+rate]
                    Xanti += Yanti / Zanti  
                
                BW_rate_shifted[rate,steps+1+rate] = BW_rate_shifted[rate,steps+rate] + V[rate-1,0] * gamma_rate * BW_rate_shifted[rate,steps+rate] * X * dt + \
                    V[rate-1,0] * BW_rate_shifted[rate,steps+rate] * gamma_rate * (BM_new[rate,steps+1] - BM_new[rate,steps])                          # Perform the Euler discretization
                BW_rate_original[rate,steps+1+rate] = BW_rate_shifted[rate,steps+1+rate] - theta_Eul[0,rate]

                
                BW_rate_anti_shifted[rate,steps+1+rate] = BW_rate_anti_shifted[rate,steps+rate] + V[rate-1,0] * gamma_rate * BW_rate_anti_shifted[rate,steps+rate] * Xanti * dt + \
                    V[rate-1,0] * BW_rate_anti_shifted[rate,steps+rate] * gamma_rate * (BManti_new[rate,steps+1] - BManti_new[rate,steps])                          # Perform the Euler discretization
                BW_rate_anti_original[rate,steps+1+rate] = BW_rate_anti_shifted[rate,steps+1+rate] - theta_Eul[0,rate]

            else: # Special case for the first backward rate
                x_axis[rate, steps+1+rate] = x_axis[rate, steps+rate] + dt                  # Add the small time step to the x-axis
               
                Tj    = x_axis[rate, rate] + tau                                            # T_j
                t     = x_axis[rate, steps+rate]                                            # Current time
                
                gamma_rate = g(Tj, t)   
                X     = Xanti = 0
                q     = int(np.ceil(((Tj * (1 / tau)) + (dt / 10))))                        # q according to Piterbarg's notation
                
                if steps == 0:                                                              # First step T_{j-1} we want the drift to come in
                    q = q-1
    
                Vol = vol
                'We dont need IC and we need a new IV'
    
                for j in range(q-1, rate+1):                                                # Perform the summation of the drift term
                    """The summation is from q(t) to j. Note that this depends on your tenor structure. I assume Piterbargs structure"""
                    """So this is different from the paper. By the definition of q(t) it makes sense. The -1 since python starts at zero."""                
                    Tj_new = x_axis[j, rate] + tau                                          # Tenor point for the summation
                    gamma = g(Tj_new, t)                                             

                    "I always use V[j,0], since the instantaneous volatility is constant I can just take the first value in column 0."
                    Y = tau * Vol * (BW_rate_original[j,steps+rate] + theta_Eul[0,j]) * gamma
                    Z = 1 + tau * BW_rate_original[j,steps+rate]
                    X += Y / Z
    
                    Yanti = tau * Vol * (BW_rate_anti_original[j,steps+rate] + theta_Eul[0,j]) * gamma
                    Zanti = 1 + tau * BW_rate_anti_original[j,steps+rate]
                    Xanti += Yanti / Zanti  
                
                BW_rate_shifted[rate,steps+1+rate] = BW_rate_shifted[rate,steps+rate] + Vol * gamma_rate * BW_rate_shifted[rate,steps+rate] * X * dt + \
                    Vol * BW_rate_shifted[rate,steps+rate] * gamma_rate * (BM_new[rate,steps+1] - BM_new[rate,steps])                                       # Perform the Euler discretization
                BW_rate_original[rate,steps+1+rate] = BW_rate_shifted[rate,steps+1+rate] - theta_Eul[0,rate]

                
                BW_rate_anti_shifted[rate,steps+1+rate] = BW_rate_anti_shifted[rate,steps+rate] + Vol * gamma_rate * BW_rate_anti_shifted[rate,steps+rate] * Xanti * dt + \
                    Vol * BW_rate_anti_shifted[rate,steps+rate] * gamma_rate * (BManti_new[rate,steps+1] - BManti_new[rate,steps])                          # Perform the Euler discretization
                BW_rate_anti_original[rate,steps+1+rate] = BW_rate_anti_shifted[rate,steps+1+rate] - theta_Eul[0,rate]

              
    # Define values after the accrual period
    for rates in range(NoOfRates):                                                                      # Loop per backward rate
        for step in range(NoOfRates + eul_steps - (NoOfRates - rates - 1), NoOfRates + eul_steps):      # Loop that starts at time R(T_j, T_{j-1}, T_j) and continues till end point
            x_axis[rates, step]   = x_axis[rates, step-1] + tau                                         # Extend x-axis after accrual period with steps of tau
            
            # Set forward-looking backward rates after accrual period
            # BW_rate[rates, step] = BW_rate[rates, step-1] 
            BW_rate_original[rates, step] = BW_rate_anti_original[rates, step] = np.nan
            
            # BW_rate_anti[rates, step] = BW_rate[rates, step-1] 
            BW_rate_shifted[rates, step] = BW_rate_anti_shifted[rates, step] = np.nan


    return BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis                                                                 # Return the forward-looking backward rates and their corresponding x-axis

# Notional   = Notional of the contract
# tau        = Difference between tenor points
# T          = Reset date of the caplet
# K          = Strike price of the caplet
# V          = Instantaneous volatility matrix
# L          = Initional forward rates
# vol        = Volatility of the first backward rate
# spotrate   = Spotrate
# theta_logE = Shift used however no shift for spotrate taken into account
"""This is because for the log-Euler method the backward and forward rates are the same
hence we do not work with the spotrate yet which is equal to the start of the first backward rates
I just define a seperate shift for that. There are also other solutions.""" 
# theta_Eul  = Shift which contains a shift for the spotrate
def cap_price_Black(Notional, tau, T, K, V, L, vol, spotrate, theta_logE, theta_Eul):  
    """Calculate the price of a caplet using Black's formula"""
    
    num_of_caps = int(T/tau)                                                                    # Number of caplets we want to validate
    cap_price = np.zeros(num_of_caps)                                                           # Save caplet prices
    
    for resetT in range(0, num_of_caps):                                                        # Apply Black's formula for every caplet                   
        helpval = (tau ** 3) / (tau ** 2)                                                       # We want the caplet price at time zero hence no maximum needed
    
        if resetT == 0:
            R = spotrate + theta_Eul[0,resetT]
            K_hat = K + theta_Eul[0,resetT]
            vsqr = (resetT * tau + 1 / 3 * helpval) * vol * vol
        else: 
            R = L[resetT-1] + theta_logE[resetT-1]
            K_hat = K + theta_logE[resetT-1]
            vsqr = (resetT * tau + 1 / 3 * helpval) * V[resetT-1, 0] * V[resetT-1, 0] 
            
               
        d1 = (np.log(R / K_hat) + (vsqr / 2)) / (np.sqrt(vsqr)) 
        d2 = d1 - (np.sqrt(vsqr))
        
        P = (1 / (1 + tau * spotrate))
        for i in range(0,resetT):      
            # Loop to calculate the value of P(0,T_n)
            P = P * (1 / (1 + tau * L[i]) )
         
         
        cap_price[resetT] = Notional * P * tau * (R * norm.cdf(d1) - K_hat * norm.cdf(d2) )
        if cap_price[resetT] / Notional < 10 ** -8:                                             # I take this as such a small value that we assume that the capprice is equal to zero
            cap_price[resetT] = 0


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
# vol        = Volatility of the first backward rate
# Spotrate   = Spotrate
# theta_logE = Shift used however no shift for spotrate taken into account
# theta_Eul  = Shift which contains a shift for the spotrate
def cap_price_MC(N, tau, K, V, L, NoOfSteps, NoOfRates, T, M, eul_steps, vol, spotrate, theta_logE, theta_Eul):
    """Calculate the price of a caplet using Monte Carlo simulation"""   
    cappriceBack = np.zeros([NoOfRates, M])                                                  # Save caplet prices for backward-looking caplet
    
    for i in range(M):
        [BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis] = Generate_Backward_Rates(NoOfSteps, NoOfRates, T, tau, V, L, eul_steps, vol, spotrate, theta_logE, theta_Eul)                # Obtain backward rates using a large time step
        for resetT in range(0, NoOfRates):                                                   # Calculate every caplet price
            # Calculate the payoffs using the forward rates from the large step method
            """We could also change the payoff using the shifted rates. This gives approximately the same answer. We should use K_hat then."""
            # if resetT == 0:
            #     K_hat = K + theta_Eul[0,resetT]
            # else:
            #     K_hat = K + theta_logE[resetT-1]
                
            payoffLS     = N * tau * max(BW_rate_original[resetT, resetT + eul_steps] - K, 0)
            payoffLSanti = N * tau * max(BW_rate_anti_original[resetT, resetT + eul_steps] - K, 0)
            
            discountLS = discountLSanti = 1
            for k in range(0,resetT+1):
                discountLS     = discountLS * (1 + tau * BW_rate_original[k,k + eul_steps])
                discountLSanti = discountLSanti * (1 + tau * BW_rate_anti_original[k,k + eul_steps])

            cappriceBack[resetT, i] = (payoffLS / discountLS + payoffLSanti / discountLSanti) * 0.5
    
    # Calculate standard errors
    se_Back  = np.sqrt(np.var(cappriceBack, axis=1, ddof=1)) / np.sqrt(M)       
    
    return np.sum(cappriceBack, axis=1) / M, se_Back

# tau        = Difference between tenor points
# V          = Instantaneous volatility matrix of the large time step method
# L          = Initial forward rates
# NoOfSteps  = Number of steps used
# NoOfRat    = Number of rates that will be simulated
# T          = End date
# M          = Number of Monte Carlo simulations
# eul_steps  = Number of Euler discretization steps
# vol        = Volatility of the first backward rate
# Spotrate   = Spotrate
# theta_logE = Shift used however no shift for spotrate taken into account
# theta_Eul  = Shift which contains a shift for the spotrate
def zero_coupon_bond(tau, V, L, NoOfSteps, NoOfRat, T, M, eul_steps, vol, spotrate, theta_logE, theta_Eul):
    """Calculate the ZCB rate analytically and using Monte Carlo simulation"""
    discount_rate = np.zeros([NoOfRat, M])                                               # Simulated ZCB
    analytZCB = np.zeros(NoOfRat)                                                        # Analytical ZCB   
    
    # Value to calculate the rate analytically 
    analytZCB[0] = (1 + tau * spotrate)                                                # L[0] / 2 = spotrat  
    for i in range(0,NoOfRat-1):
        analytZCB[i+1] = analytZCB[i] * (1 + tau * L[i])  
    
    # Perform Monte Carlo Simulation
    for i in range(M):
        [BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis] = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, vol, spotrate, theta_logE, theta_Eul)                # Obtain backward rates using a large time step
        for k in range(NoOfRat):                                                         # Loop to cover multiple ZCB       
            # Obtain the rate using Euler discretization
            pSim = pSimanti = 1                                                          # First discount rate uses spotrate
            for j in range(0, k+1):
                pSim = pSim * (1 + tau * BW_rate_original[j, j + eul_steps])
                pSimanti = pSimanti * (1 + tau * BW_rate_anti_original[j, j + eul_steps])
            discount_rate[k, i] = (1 / pSim + 1 / pSimanti) * 0.5
            
    
    # Calculate standard errors
    se_Sim = np.sqrt(np.var(discount_rate, axis=1, ddof=1)) / np.sqrt(M)
    
    # [Analytical rate, Rate using Euler, Rate using large steps]
    return 1 / analytZCB, np.sum(discount_rate, axis=1) / M, se_Sim

# NoOfSteps  = Number of steps used
# NoOfRates  = Number of rates that will be simulated
# T          = End date
# tau        = Difference between tenor points
# V          = Instantaneous volatility matrix of the large time step method
# L          = Initial forward rates
# eul_steps  = Number of Euler discretization steps
# vol        = Volatility of the first backward rate
# Spotrate   = Spotrate
# theta_logE = Shift used however no shift for spotrate taken into account
# theta_Eul  = Shift which contains a shift for the spotrate
def density_plot(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, vol, spotrate, theta_logE, theta_Eul):
    """This function makes a 3D plot of different paths for a rate and also density plots at different time intervals"""
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    NoOfLines = 150                                                             # Number of different paths you want to plot
    
    # One time calculation to obtain some sizes
    BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, vol, spotrate, theta_logE, theta_Eul)
    
    # Define the y-axis and z-axis for the plotting the rates
    zline = np.linspace(0, 0, len(x_axis[0,:]))
    yline = x_axis[NoOfRat-1,:]
    
    # Define limits for the x-axis and z-axis
    left_lim = -0.1
    right_lim = 0.1
    ax.set_xlim(left_lim, right_lim)  
    ax.set_zlim(0, 30)
    
    data = np.zeros([T,NoOfLines])                                              # Save data to make density plots
       
    for i in range(NoOfLines):
        # Plot different paths
        BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, vol, spotrate, theta_logE, theta_Eul)
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
    T           = 3                                                                 # Time horizon
    eul_steps   = 50                                                                # Euler discretization steps between two tenor points
    rate_type   = 'negative'                                                        # Start with positive/negative rates
    
    
    NoOfSteps   = int(T / tau)                                                      # Number of time-steps                                                    
    NoOfRat     = int(T / tau)                                                      # Number of backward rates we want to generate

    # Note that normally you will obtain these using calibration, now I simulate them randomly
    [V, vol]    = insVol(NoOfSteps, NoOfRat)                                        # Instantaneous volatility
    L           = inFor(NoOfRat, rate_type)                                         # Initial rates
    spotrate    = -0.06                                                             # Spotrate
    [theta_logE, theta_Eul] = theta(NoOfRat, L, spotrate)                           # Obtain shift valeus

    # Obtain the forward-looking backward rates and their corresponding x-axis
    BW_rate_original, BW_rate_anti_original, BW_rate_shifted, BW_rate_anti_shifted, x_axis = Generate_Backward_Rates(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, vol, spotrate, theta_logE, theta_Eul)
        
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
        ax.plot(x_axis[i,:], BW_rate_shifted[i,:])
        
    plt.title('Simulated backward-looking forward rates')
    plt.ylabel('Rate (%)')
    plt.xlabel('Time (years)')
    plt.xticks(locs,labels, rotation='45') 
    plt.grid()

    # "Validating caplet prices"
    # Notional = 1000000                                                                 # Notional
    # K = -0.02                                                                        # Strike price
    # M = 1000                                                                         # Number of Monte Carlo simulations

    # [capLS, se_LS] = cap_price_MC(Notional, tau, K, V, L, NoOfSteps, NoOfRat, T, M, eul_steps, vol, spotrate, theta_logE, theta_Eul)         # Obtain the simulated caplet prices
    # capBlack = cap_price_Black(Notional, tau, T, K, V, L, vol, spotrate, theta_logE, theta_Eul)   
    
    # # Generate nice ouput
    # colname = list()
    # for i in range(NoOfRat):
    #     name = '{}{}'.format('Cap', '(T_' + str(i) + ',' + 'T_' + str(i+1) + ')')
    #     colname.append(name)
  
    # table = zip(colname, capBlack, capLS, 100 * abs(capLS - capBlack) /  capBlack, se_LS)
    # header = ['Analytical Price', 'Simulated price', 'Error (%)', 'Standard\n error']  
    
    # print(tabulate(table, headers = header, tablefmt="fancy_grid"))
    
    # "Valiating zero-coupon bond prices"
    # M = 500
    
    # [anrate, sim_rate, se_Sim] = zero_coupon_bond(tau, V, L, NoOfSteps, NoOfRat, T, M, eul_steps, vol, spotrate, theta_logE, theta_Eul)      # Obtain the differenct calculatd zero-coupon rates
    
    # colname = list()
    # for i in range(NoOfRat):
    #     name = '{}{}'.format('P', '(T_' + str(i+1) + ')')
    #     colname.append(name)
  
    # table = zip(colname, anrate, sim_rate, 100 * abs(sim_rate - anrate) / anrate, se_Sim)
    # header = ['Analytical Price', 'Simulated price', 'Error (%)', 'Standard error']
    
    # print(tabulate(table, headers = header, tablefmt="fancy_grid"))
    
    "Make density plot"
    density_plot(NoOfSteps, NoOfRat, T, tau, V, L, eul_steps, vol, spotrate, theta_logE, theta_Eul)
      
# Time the program           
start = timeit.default_timer()   
print('Starting the function') 
mainCalculation()
stop = timeit.default_timer() 
print('Time: ', stop - start, '\n')