import random as rd
import numpy as np
import yaml
import torch

from collections import deque
from model import Linear_QNet, QTrainer

RL_SIG = {'1' : 'DO_STH' , '2' : 'DO_NTH'}

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Loss Rate

class RL_Agent:

    def __init__(self):
        self.lt = 0 # loop time
        self.rn = 0 # randomness, higher -> more random
        self.dr = 0.9 # discount rate
        self.epsilon = 0.00001 # | Qnew - Qold | < ε, Qnew = Qmax
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(10, 256, 9)
        self.trainer = QTrainer(self.model, lr=LR, dr=self.dr)
        
     
        
    # def get_state(self, MSEArray):
    #     print(MSEArray)
    #     s = torch.tensor(MSEArray, dtype=torch.float)
    #     print(s)
    #     state = s.norm(dim=-1, p=2)
    #     print("get_state:", state)
    #     return state
    
    def get_action(self, state):
        self.rn = 80 - self.lt
        
        # assume final action = [CPU, RAM, REPLICA]
        #     [+V, -V, 0, | +V, -V, 0, | +1, -1, 0]
        # +V means increase the usage of CPU, same as RAM
        # +1 means add 1 replica,   
        # no matter +V or +1, the value in each block should be 1 (instead of V)
        # for the second bit, it's positive value showing the decrease usage of CPU/RAM
        # if there's nothing change, the third bit should be 1 and first/second bit should be 0
        
        final_action = np.zeros(9)
        
        if rd.randint(0, 150) < self.rn:
            for i in range(3):
                rint = rd.randint(-1,1)
                
                print("action:", rint)
                
                if rint == 1: final_action[i*3] = 1
                elif rint == -1: final_action[(i*3)+1] = 1
                else: final_action[(i*3)+2] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            
            # print("state0:",state0)
            
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        print("final:", final_action)
        return final_action
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state)) # popleft if MAX_MEMORY is reached
        
        # print(self.memory)
        
        
    def train_long_memory(self):
        print(len(self.memory))
        if len(self.memory) > BATCH_SIZE:
            mini_sample = rd.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states = zip(*mini_sample)
        
        # print("long memory states:",states)
        
        self.trainer.train_step(states, actions, rewards, next_states)
        
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state):
        self.trainer.train_step(state, action, reward, next_state)

def gau_generator(sample):
    gr = np.random.normal(0.5,0.2,sample)
    return gr  

def loop(loop_t):
    agent = RL_Agent()
    reward = [0,0,0]
    # read the initial state
    
    
    # initialize value of replica
    with open('./nginx.yml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    # print("initial", result['spec'])
    
    for i in range(loop_t):
        # create random signal
        
        # sig = gen_rl_sig()
        
        # print(sig," at: ",i)
        
        # if sig == 'DO_STH': 
        #     result['spec']['replicas'] = rd.randint(1,10)
        #     # print(replica, " : ", result['spec'])
            
        #     if replica != result['spec']['replicas']:
                
        #         with open('./nginx.yml', 'w', encoding='utf-8') as f:
        #             yaml.dump(data=result,stream=f,allow_unicode=True)
        #         replica = result['spec']['replicas']
        #     else: print("same value of replica compare to previous loop")
        
        
        
        # create random state (for now)
        # get old state
        state_old = gau_generator(10)
        
        # state_old = agent.get_state(MSE_old)
        
        
        # get action
        final_action = agent.get_action(state_old)
        
        # print("final action:",final_action)
        
        # send action to testbed
        # result['spec']['cpu'] += final_action[0]
        # if(result['spec']['cpu'] > 8) : result['spec']['cpu'] = 8
        # if(result['spec']['cpu'] < 0) : result['spec']['cpu'] = 0
        
        # result['spec']['ram'] += final_action[1]
        # if(result['spec']['ram'] > 64) : result['spec']['ram'] = 64
        # if(result['spec']['ram'] < 0) : result['spec']['ram'] = 0
        
        # result['spec']['replicas'] += final_action[2]
        # if(result['spec']['replicas'] > 8) : result['spec']['replicas'] = 8
        # if(result['spec']['replicas'] < 1) : result['spec']['replicas'] = 1
        
        # print(result['spec'])
        # with open('./nginx.yml', 'w', encoding='utf-8') as f:
        #     yaml.dump(data=result,stream=f,allow_unicode=True)
        
        # for now new state will be generated with random value
        state_new = gau_generator(10)
        
        # get new state
        # state_new = agent.get_state(MSE_new)
        
        # for i in range(len(state_new)):
        #     reward[i] = (1/state_new[i]) - (1/state_old[i])
        
        # reward = 1 / (vector distance)
        reward_new = np.linalg.norm(np.array(state_new))
        reward_old = np.linalg.norm(np.array(state_old))
        print("state new: ", state_new)
        print("reward: ", reward_new)
        
        # train short memory
        agent.train_short_memory(state_old, final_action, reward_new, state_new)
        
        # remember all possible state
        agent.remember(state_old, final_action, reward_new, state_new)
        
        # train long memory
        # agent.train_long_memory()
        
        # save the record
        if reward_new > reward_old:
            agent.model.save()
        
        agent.lt += 1
        
        
                
        
        
loop(1000)