import numpy as np
import pandas as pd
import pickle as pkl
import random


def purePeriod(weights,periods,phases,t):

    signal = 0
    assert (len(weights)==len(periods)) and (len(phases)==len(periods))
    for i in range(len(weights)):
        signal+=weights[i]*np.sin((2*np.pi/periods[i])*t+phases[i])
    
    return signal,signal,np.array([0 for _ in range(len(signal))])

def poly(t):
    # weights = np.array([random.uniform(-0.0000001,.0000001) for _ in range(3)])
    weights = np.array([random.uniform(-0.0000000001,.0000000001) for _ in range(3)])
    res = np.array([1 for _ in range(len(t))])*weights[0]
    res += t*weights[1]
    res += np.array([i**2 for i in t])*weights[2]

    return res
    
def g_linear(k,b,t):
    return k*t+b


def full(weights,periods,phases,k,b,t,is_linear,numOfBase):



    if is_linear:
        nt = g_linear(k,b,t)
    else:
        nt = poly(t)
    
    if numOfBase==0:
        all = nt
        return all, 0, nt

    signal = 0
    assert (len(weights)==len(periods)) and (len(phases)==len(periods)) 
    for i in range(len(weights)):
        signal+=weights[i]*np.sin((2*np.pi/periods[i])*t+phases[i])

    all = signal + nt +np.random.rand(t.shape[0])*0.01

    return all,signal,nt




def signal_sim(args,is_full,is_linear,numOfBase,input_len,out_len,numOfSamples,granularity):
    

    if numOfBase == 30:
        periods = np.array([i for i in range(1,1+numOfBase)])
    else:
        periods = np.random.choice(a=100, size=numOfBase, replace=False, p=None)+1
    print ("Sampled periods: {}".format(len(periods)))

    weights = np.array([random.uniform(-1,1) for _ in range(len(periods))])
    phases =  np.array([random.uniform(-1,1) for _ in range(len(periods))])
    k = args.k
    b = args.b

    t = np.array([0+i*granularity for i in range(numOfSamples+input_len+out_len-1)])

    if not is_full:
        print ("purePeriod")
        signal,p_sig,np_sig = purePeriod(weights,periods,phases,t)
    else:
        print ("full")
        signal,p_sig,np_sig = full(weights,periods,phases,k,b,t,is_linear,numOfBase)


    
    signal_df = pd.DataFrame(data={"date":[0 for _ in range(len(signal))],"V":signal,"time":t})

    data = {
        "weights":weights,
        "periods":periods,
        "phases":phases,
        "p_sig":p_sig,
        "np_sig":np_sig
    }

    # with open("../result/simulation_"+str(numOfBase)+"_"+str(numOfSamples)+"_"+str(granularity)+".pkl","wb") as f:
    #     pkl.dump(data,f)

    signal_df.to_csv("../data/simulation_"+str(numOfBase)+"_"+str(numOfSamples)+"_"+str(granularity)+".csv",index=None)

    return


if __name__ == "__main__":
    print (signal_sim(False,2,3,2,5,.1))