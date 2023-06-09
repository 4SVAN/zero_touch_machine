import random as rd
import numpy as np
import yaml

RL_SIG = {'1' : 'DO_STH' , '2' : 'DO_NTH'}

def gau_generator(sample):
    gr = np.random.normal(0.5,0.2,sample)
    return gr

def gen_rl_sig():
    x = gau_generator(1)
    if x > 0.5: return RL_SIG['1']
    else: return RL_SIG['2']

def loop(loop_t):
    # initialize value of replica
    with open('./nginx.yml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    replica = result['spec']['replicas']
    
    for i in range(loop_t):
        # create random signal
        x = gau_generator(10)
        sig = gen_rl_sig()
        
        print(sig," at: ",i)
        
        if sig == 'DO_STH': 
            result['spec']['replicas'] = rd.randint(1,10)
            print(replica, " : ", result['spec'])
            
        if replica != result['spec']['replicas']:
            with open('./nginx.yml', 'w', encoding='utf-8') as f:
                yaml.dump(data=result,stream=f,allow_unicode=True)
            replica = result['spec']['replicas']
                
        
        
loop(5)

