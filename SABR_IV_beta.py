import numpy as np
from scipy.stats import norm
import scipy.optimize as optimize
import pandas as pd
import matplotlib.pyplot as plt

rho   = 0
shift = 0.03

# NoOfRates      = Number of rates
# Initial_market = Initial market rates
def inFor(NoOfRates, initial_market): 
    """Generate the initial forward rates."""      
    L = initial_market[:NoOfRates].to_numpy()
    return L 

def Black(Notional, tau, K, L, vB, P):  
    """Calculate the price of a caplet using Black's formula"""    
    R = L
    K_hat = K
    
    d1 = (np.log(R / K_hat) + (vB / 2)) / (np.sqrt(vB)) 
    d2 = d1 - (np.sqrt(vB))
     
    cap_price = Notional * P * tau * (R * norm.cdf(d1) - K_hat * norm.cdf(d2) )
    return cap_price

def ImpliedVolatility_back(marketPrice, tau, K, L, sigmaInitial, P, Notional, resetT):
    func = lambda sigma: np.power(Black(Notional, tau, K, L, sigma, P) - marketPrice, 1.0)
    vB = optimize.newton(func, sigmaInitial, tol=1e-10, maxiter=200)
  
    impliedVol = np.sqrt(vB / (resetT * tau))
    
    return impliedVol, vB

# Notional   = Notional of the contract
# dif        = Difference between tenor points
# strike     = Strike price of the caplet
# L          = Initional forward rates
# v          = Volatility parameter
# IV         = Instantaneous volatility
def cap_price_Black_SABR(Notional, dif, strike, L, v, IV, resetT, rates, beta):  
    """Calculate the price of a caplet using Black's formula"""                                                      # Save caplet prices
    q           = 1              
       
    # Specific caplet parameters
    tau_0     = resetT * dif
    tau_1     = tau_0 + dif
    R         = L
    alpha_cap = IV
    K         = strike

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
    
    if strike != L:
        z   = np.sqrt(v_hat) / np.sqrt(alpha_hat) * (R * K) ** ((1-beta) / 2) * x
        ksi = np.log((np.sqrt(1 - 2 * rho_hat * z + z ** 2) + z - rho_hat) / (1 - rho_hat))
        
        # sigma-hagan               
        a    = np.sqrt(alpha_hat) / (R * K) ** ((1 - beta) / 2)
        b    = (1 + (1 - beta) ** 2 / 24 * x ** 2 + (1 - beta) ** 4 / 1920) ** -1
        c    = z / ksi
        d    = ((1 - beta) ** 2 / 24) * (alpha_hat / (R * K) ** (1 - beta))
        e    = 0.25 * (rho_hat * beta * np.sqrt(v_hat) * np.sqrt(alpha_hat)) / (R * K) ** ((1 - beta) / 2)
        f    = v_hat * ((2 - 3 * rho_hat ** 2) / 24)  
        vsqr = (a * b * c * (1 + (d + e + f) * tau_1)) ** 2
    else:
        a = np.sqrt(alpha_hat) / (R ** (1 - beta))
        b = ((1 - beta) ** 2 / 24) * (alpha_hat / (R ** (2 - 2 * beta)))
        c = 0.25 * ((rho_hat * beta * np.sqrt(alpha_hat) * np.sqrt(v_hat)) / (R ** (1 - beta)))
        d = ((2 - 3 * rho_hat ** 2) / 24) * v_hat
        vsqr = (a * (1 + (b + c + d) * tau_1)) ** 2        
    
    
    d1 = (x + (vsqr / 2) * tau_1) / (np.sqrt(vsqr)  * np.sqrt(tau_1)) 
    d2 = d1 - (np.sqrt(vsqr) * np.sqrt(tau_1))    

    P = 1 / (1 + dif * (-0.0049754))
    for i in range(0,resetT):
        P = P * (1 / (1 + dif * (rates[i+1])))

    cap_price = Notional * P * dif * (R * norm.cdf(d1) - K * norm.cdf(d2) )
    if cap_price / Notional < 10 ** -8:                                             # I take this as such a small value that we assume that the capprice is equal to zero
        cap_price = 0
    
    return cap_price, vsqr * tau_1, P

tau         = 0.25                                                              # Difference between tenor points
T           = 40                                                                 # Time horizon
eul_steps   = 64                                                                # Euler discretization steps between two tenor points    

NoOfSteps   = int(T / tau)                                                      # Number of time-steps                                                    
NoOfRat     = int(T / tau)                                                      # Number of backward rates we want to generate

# Obtain the correct data (Initial rates, Instantaneous volatilities)
df = pd.read_excel (r'C:\Users\hackt\Documents\Thesis Rabobank\Python\Caplets_3Months.xlsx')

help1 = df.iloc[17:,:]
data = help1.iloc[:,[1,2,5,7]]
data.reset_index(drop=True, inplace=True)
data.columns = ['Date', 'CapletForward', 'DF', 'IV']

rates         = inFor(NoOfRat, data.CapletForward)  / 100                                        # Initial rates
Notional  = 10000
v         = 0.1
disc      = 20

resetT = 16 # 4 years
L_neg  = rates[resetT-1]
vol    = data.IV[resetT-1]

L      = L_neg + shift
strike = np.linspace(0.25*L, 4*L, disc)


price   = np.zeros(len(strike))
sigmaIn = np.zeros(len(strike))

IV_back = np.zeros(len(strike))
vB      = np.zeros(len(strike))


linestyles = ['v','h', '+', 'd', 'o', 's']
pos        = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(len(pos)):
    beta = pos[i]
    
    for j in range(len(strike)):
        [price[j], sigmaIn[j], P] = cap_price_Black_SABR(Notional, tau, strike[j], L, v, vol, resetT, rates, beta)

    for k in range(len(strike)):
        IV_back[k], vB[k] = ImpliedVolatility_back(price[k], tau, strike[k], L, sigmaIn[k], P, Notional, resetT)
    
    
    # plt.plot(strike, price, label='{}'.format(pos[i]), marker=linestyles[i])
    plt.plot(strike, vB, label='{}'.format(pos[i]), marker=linestyles[i])
    
    
plt.title('Implied volatility curve (Denoted as vB)')
# plt.title('Caplet price for different strike prices')
plt.xlabel('Shifted strike')
plt.ylabel('IV (%)')
# plt.ylabel('Price (bps)')
plt.axvline(x = L, color='k', linestyle='--', label='ATM')
plt.grid()
plt.legend(title='Beta')