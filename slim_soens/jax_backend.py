import numpy as np
import jax.numpy as jnp
import jax

# import matplotlib.pyplot as plt
import time
from system_functions import get_jj_params

# plt.style.use('seaborn-v0_8-muted')
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
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

@jax.jit
def transfer_function(flux,signal):
    return flux-flux_threshold+(.8-signal)
    # return  np.piecewise(signal, [signal < 0, signal >= 0.1675], [0, flux-1.675+(.8-signal)]) 

@jax.jit
def step(t,flux,signal,dalbe,debe,inpt,inpt_adj,adj):
    # flux = flux + inpt[:,t]*inpt_adj

    r_fq = transfer_function(flux,signal)
    # r_fq = r_fq.at[r_fq<flux_threshold].set(0) 

    signal = signal * dalbe + debe * r_fq

    # flux = jnp.dot(signal,adj)
    flux = jnp.matmul(signal,adj)
    # flux = jax.lax.dot(signal,adj)
    
    return flux, signal

# @jax.jit
def run_vectorized_simulation(
    T,flux,signal,
    adj,inpt,inpt_adj,
    d_tau,ib,tau,beta,alpha,flux_threshold
    ):
    # fluxes = []
    # signals = []
    dalbe = (1-d_tau*alpha/beta) 
    debe = d_tau/beta
    # flux = flux + inpt[:,0]*inpt_adj
    for t in range(T):
        
        flux, signal = step(t,flux,signal,dalbe,debe,inpt,inpt_adj,adj)

# print(jax.default_backend())
# from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)


N = 20000
T = 100


adj         = jnp.array((np.multiply(np.random.randint(2, size=(N,N)),np.random.rand(N,N))).astype(np.float32))

inpt_signal = jnp.array(np.sin(np.arange(-np.pi, np.pi, 2*np.pi/T)).astype(np.float32)*.5+1)
inpt        = jnp.array(np.array([inpt_signal for _ in range(N)]).astype(np.float32))
inpt_adj    = jnp.array(np.random.randint(2,size=(N,)).astype(np.float32))


flux = jnp.array(np.zeros(N).astype(np.float32))
signal = prev_signal = jnp.array(np.zeros(N).astype(np.float32))

t1 = time.perf_counter()

run_vectorized_simulation(
    T,flux,signal,
    adj,inpt,inpt_adj,
    d_tau,ib,tau,beta,alpha,flux_threshold
    )

t2 = time.perf_counter()

print(f"Time to run the simulation: {t2-t1}")



flux = jnp.array(np.zeros(N).astype(np.float32))
signal = prev_signal = jnp.array(np.zeros(N).astype(np.float32))

t1 = time.perf_counter()

run_vectorized_simulation(
    T,flux,signal,
    adj,inpt,inpt_adj,
    d_tau,ib,tau,beta,alpha,flux_threshold
    )

t2 = time.perf_counter()


print(f"Time to run the simulation: {t2-t1}")
# # print(flux,signal)