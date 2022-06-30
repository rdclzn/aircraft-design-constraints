# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:56:29 2022

@author: Emanuel Mendes & Bruna Milione
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


#diametro fuselagem = 1.55 m

def taper_function(lbd):
    f = 0.0524*lbd**4 - 0.15*lbd**3 + 0.1659*lbd**2 -0.0706*lbd + 0.0119
    return f

def taper_ratio(AR, b, Cr):
    lambd = (2*b/(AR*Cr)) - 1
    return lambd

def oswald_number(AR, Cr, df, b, Cd0):
    #Kroo-based method
    #Proposed by M. Nita, D. Scholz
    
    #Zero sweep, trapezoidal wing
    e_theo = 1/(1 + taper_function(taper_ratio(AR,b,Cr) - 0.357)*AR)
    
    #Velocidade do som ~350 kph, velocidade maxima na tabela comparativa ~190 kph
    #Regime incompressível, logo:
    k_em = 1
    
    k_ef = 1 - 2*(df/b)**2
    
    Q = 1/(e_theo*k_ef)
    P = 0.38*Cd0
    e = k_em/(Q + P*np.pi*AR)
    return e

def Cf(Re,laminar):
    if laminar:
        Cf = 1.328/(Re**0.5)
    else:
        #Cf = 0.455/(np.log10(Re)**2.58 + 0.01)
        Cf = 0.455/(np.log10(1826642)**2.58 + 0.01)
    return Cf

def CD0(V,rho,b,AR,df,tc):
    d = b/AR
    Re = rho*V*d/(1.918E-5)
    length = 3
    
    #Surface roughness, characteristic length
    #smooth paint = 0.0634
    #sheet metal = 0.0405
    #polished metal = 0.0152
    #Smooth composite = 0.0052
    k = 3E-6 #chute educado -> roughness
    Re_cutoff = 38.21*(0.0052/k)**1.053
    Re = min(Re,Re_cutoff)
    cf = Cf(Re,False)
    
    S_ref_wing = (b**2)/AR
    S_ref_fus = np.pi*df*length
    S_ref = S_ref_wing + S_ref_fus
    S_wet_wing = S_ref_wing*(1.977 + 0.52*tc)
    e = np.sqrt(1 - (df**2)/(length**2))
    S_wet_fus = 2*(np.pi*df**2)*(1 + (length/df)*np.arcsin(e)/e)
    S_wet = S_wet_wing + S_wet_fus
    
    cd0 = cf*(S_wet/S_ref)
    return cd0

def W_motor(P):
    P_unit = P/2
    W = P_unit/5000  #5000W/kg for an EMRAX
    if (W<=7.2):
        return 7.2*2
    else:
        if (W<=9.3):
            return 9.3*2
        else:
            if (W<=20.3):
                return 20.3*2
            else:
                if (W<=41.5):
                    return 41.5*2

def W_empty(AR):
    w = 22*AR
    if(w<180):
        return 180
    else:
        return w

def CD(V,rho,b,AR,df,tc, Cr, Cl):
    Cd0 = CD0(V,rho, b, AR, df, tc)
    e = oswald_number(AR,Cr,df,b,Cd0)
    Cd = Cd0 + (Cl**2)/(np.pi*AR*e)
    return Cd

def Restricoes(x):
    #Atribuindo variáveis independentes
    P = x[0]
    W_batt = x[1]
    W_load = x[2]
    AR = x[3]
    b = x[4]
    Cr = x[5]
    Cl = x[6]
    V = x[7]
    Cl_climb = x[8]
    V_climb = x[9]
    W = 600*9.81
    rho = 0.7423 #ISA @3000m Altitude
    rho_climb = 1.111 #@ISA @1000ft Altitude +20°C
    df = 1.55
    tc = 0.1
    
    if P > 80:
        P = 80
    
    print(f'potencia: {P}')
    P_max = W_motor(P)*(9/5) #9000W/kg peak 5000W/kg sustained
    
    Cd_climb = CD(V_climb,rho_climb, b, AR, df, tc, Cr, Cl_climb)
    h_dot = P_max/W - (Cd_climb/Cl_climb)*V_climb
    
    Cd = CD(V,rho, b, AR, df, tc, Cr, Cl)
    S = (1/AR)*(b**2)
    Vc = (2*P*0.8/(rho*S*Cd))**(1/3)  #eficiencia propulsiva = 0.8
    
    #Atribuindo restrições
    y1 = h_dot
    y2 = h_dot - Vc
    y3 = Vc - V
    y4 = Cr
    y5 = Cl
    y6 = V
    y7 = W_motor(P) + W_load + W_batt + W_empty(AR) - W
    y8 = b
    y9 = P
    return ([y1,y2,y3,y4,y5,y6,y7,y8,y9])

#lower boundary
lb = [20, -np.inf, 0, 0, 0, 23.3, 0, 0, 0]

#upper boundary
ub = [200, 0, 0, 11, np.inf, np.inf, 0, 11, 415000]

#initial guess
x0 = [25, 220, 212.8, 8, 11, 1, 0.25, 55, 0.35, 35]



##################################
# objective function
obj_fun = lambda x: -(x[2] * x[7]**2) 

cons = {'type': 'ineq', 'fun': Restricoes}

# Result
result = minimize(obj_fun, x0, method = 'SLSQP', constraints = cons)
print(result)

