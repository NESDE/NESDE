import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from NESDE import HyperNESDE
import torch
import pickle
import os
dir_path = os.path.dirname(os.path.realpath(__file__))



class AntiCoagulantEnv(gym.Env):

    def __init__(self, device, dt=1.0, max_episode_len=6000, target_aPTT=[40,60]):
        with open(dir_path + "/nesde_hyperparameters.pkl", 'rb') as f:
            nesde_kwargs = pickle.load(f)
        nesde_kwargs['device'] = torch.device(device)
        self.nesde = HyperNESDE(**nesde_kwargs).to(device)
        self.nesde.device = device
        self.nesde.load_state_dict(torch.load(dir_path + "/nesde_weights.pt",map_location=device))
        self.dist = torch.distributions.Normal
        self.action_space = spaces.Box(
            low=0.0,
            high=30000.0, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=200.0, shape=(1,),
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None
        self.prev_S = None
        self.prev_S_var = None
        self.dt = dt
        self.max_episode_len = max_episode_len
        self.curr_weight = None
        self.target_aPTT = target_aPTT
        self.S_mask = torch.zeros(self.nesde.n + 2*self.nesde.m,device=device).type(torch.bool)
        self.S_mask[0] = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, dt=None):
        if dt is None:
            dt = self.dt
        err_msg = "%r (%s) invalid" % (action, type(action))
        action = action / self.curr_weight
        assert self.action_space.contains(action), err_msg
        with torch.no_grad():
            St, St_var = self.nesde(self.prev_S, self.prev_S_var, dt*torch.ones(1,1,device=self.nesde.device), U=torch.Tensor(action).to(self.nesde.device).view(1,1))
        # sample
        dist = self.dist(St[...,0],torch.clip(St_var[...,0,0],1e-1))
        Sample = torch.clip(dist.sample(),max=125.0,min=18.0)
        self.prev_S, self.prev_S_var = self.nesde.conditional_dist(St, St_var, self.S_mask, Sample)

        self.curr_time = self.curr_time + dt
        done = False
        self.state = Sample.view(1).cpu().numpy()
        if self.curr_time > self.max_episode_len:
            done = True
        if not done:
            if (self.state < self.target_aPTT[1]) and (self.state > self.target_aPTT[0]):
                reward = 1.0
            elif self.state >= self.target_aPTT[1]:
                reward = self.target_aPTT[1] - self.state
            elif self.state <= self.target_aPTT[0]:
                reward = self.state - self.target_aPTT[0]
        else:
            reward = 0.0
        return self.state , reward, done, {}

    def reset(self):
        self.prev_S = None
        self.prev_S_var = None
        # sample context
        with torch.no_grad():
            self.curr_weight = self.nesde.sample_context()
            St, St_var = self.nesde(self.prev_S, self.prev_S_var, torch.zeros(1, 1, device=self.nesde.device),
                                    U=torch.zeros(1,1,device=self.nesde.device))
        dist = self.dist(St[...,0],St_var[...,0,0])
        Sample = torch.clip(dist.sample(),max=125.0,min=18.0)
        self.prev_S, self.prev_S_var = self.nesde.conditional_dist(St, St_var, self.S_mask, Sample)
        self.state = Sample.view(1).cpu().numpy()
        self.curr_time = 0.0
        return self.state

    def render(self, mode='human'):
        print("Not Implemented!")

    def close(self):
        del self.nesde

if __name__ == "__main__":

# usage example:
    device = torch.device('cpu')
    env = AntiCoagulantEnv(device, dt=1.0, max_episode_len=6000, target_aPTT=[40,60])
    sample = env.action_space.sample()
    state = env.reset()
    num_steps = 120003
    for i in range(num_steps):
        state, reward, done, info = env.step(0.1*np.ones_like(sample))
        if done:
            state = env.reset()

