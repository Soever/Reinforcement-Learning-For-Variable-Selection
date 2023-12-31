from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import logging
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def save_sample(transition_dict,state,action,next_state,reward,done,mask):
    transition_dict['states'].append(state)
    transition_dict['actions'].append(action)
    transition_dict['next_states'].append(next_state)
    transition_dict['rewards'].append(reward)
    transition_dict['dones'].append(done)
    transition_dict['masks'].append(mask)


def train_on_policy_agent(env, agent, num_episodes,epochs):
    return_list = []
    bestr2_list =[]
    best_state_list =[]

    for i in range(epochs):
        with tqdm(total=int(num_episodes/epochs), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/epochs)):

                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],'masks':[]}
                state = env.reset()
                episode_return = env.best_R2
                done = False
                while not done:
                    scale_state  = state/env.ub# 防止输入差别过大神经网络出现错误，因此对输入做归一化
                    invalid_action_mask = env.get_action_mask()
                    action = agent.take_action(scale_state,torch.from_numpy(invalid_action_mask).to(agent.device))
                    next_state, reward, done,info= env.step(action)
                    save_sample(transition_dict,scale_state,action,next_state/env.ub,reward,done,invalid_action_mask)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                bestr2_list.append(env.best_R2)
                best_state_list.append(env.best_state)
                agent.update(transition_dict)
                if (i_episode+1) % epochs == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/epochs * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:]),'R2': '%.3f' % np.mean(bestr2_list[-10:])})
                pbar.update(1)

    return return_list,bestr2_list,best_state_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

