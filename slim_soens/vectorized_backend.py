import numpy as np
import matplotlib.pyplot as plt
import time
from system_functions import get_jj_params
from numba import jit


plt.style.use('seaborn-v0_8-muted')
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
tau = 150
beta   = 2*np.pi*1e3
ib = 1.8
flux_threshold = 0.1675
jjparams = get_jj_params()
d_tau = jjparams['t_tau_cnvt']
tau_di = tau * 1e-9  
Ic = 100 * 1e-6
Ldi = 6.62606957e-34/(2*1.60217657e-19)*beta/(2*np.pi*Ic)
rdi = Ldi/tau_di
alpha = rdi/2.565564120477849


N = 20000
T = 100

# N = 3
# T = 100
# inpt_adj = np.array([1,2,0])
# adj = np.array([
#     [0,0,1],
#     [0,0,1],
#     [0,0,0]
# ])


adj = np.multiply(np.random.randint(2, size=(N,N)),np.random.rand(N,N))

inpt_signal = np.sin(np.arange(-np.pi, np.pi, 2*np.pi/T))*.5+1
inpt = np.array([inpt_signal for _ in range(N)])
inpt_adj = np.random.randint(2,size=(N,))




# @jit(nopython=True)
def transfer_function(flux,signal):
    return flux-flux_threshold+(.8-signal)
    # return  np.piecewise(signal, [signal < 0, signal >= 0.1675], [0, flux-1.675+(.8-signal)]) 

# @jit(nopython=True)
def step(t,flux,signal,alpha,beta,d_tau,inpt,inpt_adj,adj):
    flux = flux + inpt[:,t]*inpt_adj

    r_fq = transfer_function(flux,signal)
    r_fq[r_fq<flux_threshold] = 0

    signal = signal * (1-d_tau*alpha/beta) + (d_tau/beta) * r_fq

    flux = np.dot(signal,adj)
    return flux, signal

# @jit(nopython=True)
def run_vectorized_simulation(
    T,flux,signal,
    adj,inpt,inpt_adj,
    d_tau,ib,tau,beta,alpha,flux_threshold
    ):
    # fluxes = []
    # signals = []
    for t in range(T):
        
        flux, signal = step(t,flux,signal,alpha,beta,d_tau,inpt,inpt_adj,adj)
        # flux = flux + inpt[:,t]*inpt_adj

        # r_fq = transfer_function(flux,signal)
        # r_fq[r_fq<flux_threshold] = 0

        # signal = signal * (1-d_tau*alpha/beta) + (d_tau/beta) * r_fq


        # flux = np.dot(signal,adj)

        # fluxes.append(flux)
        # signals.append(signal)

    

# plt.figure(figsize=(8,4))
# for n in range(N):

#     plt.plot(np.array(fluxes)[:,n],'--',c=colors[n], label=n)
#     plt.plot(np.array(signals)[:,n], c=colors[n])

# plt.legend()
# plt.show()



flux = np.zeros(N)
signal = prev_signal = np.zeros(N)

t1 = time.perf_counter()

run_vectorized_simulation(
    T,flux,signal,
    adj,inpt,inpt_adj,
    d_tau,ib,tau,beta,alpha,flux_threshold
    )

t2 = time.perf_counter()

print(f"Time to run the simulation: {t2-t1}")



flux = np.zeros(N)
signal = prev_signal = np.zeros(N)

t1 = time.perf_counter()

run_vectorized_simulation(
    T,flux,signal,
    adj,inpt,inpt_adj,
    d_tau,ib,tau,beta,alpha,flux_threshold
    )

t2 = time.perf_counter()


print(f"Time to run the simulation: {t2-t1}")
# # print(flux,signal)