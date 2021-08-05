import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE_2 = 50000   # memory 2
BUFFER_SIZE_3 = 50000 # replay buffer size
BATCH_SIZE = 16         # minibatch size
BATCH_SIZE_2 = 32   # FOR NEW MEMORY
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 2        # how often to update the network
total_reward = 0 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alpha_start=0.05
alpha_end= 1
alpha_rate=0.995

class Agent():

    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.q = np.zeros((state_size,action_size))
        # Replay memory
        self.memory1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        #self.memory2 = Replay2(action_size, BUFFER_SIZE_3, BATCH_SIZE, seed)
        self.memory3 = ReplayBuffer(action_size, BUFFER_SIZE_3, BATCH_SIZE, seed)
        self.total_reward = 0
        self.mem1_learn_counter = 0
        self.mem3_learn_counter = 0
        self.mem_counter1 = 0 
        self.mem_counter2 = 0 
        
        self.lenmem1 = 0
        #self.lenmem2 = 0
        self.lenmem3 = 0
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        
        self.memory1.add(state, action, reward, next_state, done)
        #self.memory2.add(state, action, reward, next_state, done,self.q)
        
        # if  self.mem_counter % 10 == 0 and len(self.memory2) > (BATCH_SIZE_2): 
            
        #     pstate, paction, preward, pnext_state, pdone,self.q  = self.memory2.popleft()
            
            
        #     #np.min(self.q)
        #     # self.q = self.q.cpu().data.numpy()
        #     #print(type(self.q))
        #     #[print(self.q[i]) for i in range(len(self.memory2))]
        #     # index = random.randrange(1,BATCH_SIZE)
        #     # pstate, paction, preward, pnext_state, pdone = self.memory2.pop(index)
        #     #self.total_reward  += preward

        #     avg_reward = self.total_reward / (len(self.memory2) * 0.01 )
            
        #     #TO DO 
        #     if (avg_reward < 5):
                
        #         self.memory3.add(pstate, paction, preward, pnext_state, pdone)
      
        self.mem_counter1 += 1
        
        
        self.lenmem1 =  self.mem_counter1
        #self.lenmem2 = len(self.memory2)
 
        self.experience_replay()
    
    # CHOOSE TO REPLAY
    def experience_replay(self):
        
        alpha = alpha_start 

        if random.random() > alpha:

            if len(self.memory1) > BATCH_SIZE:
                experiences = self.memory1.sample()
                self.learn(experiences, GAMMA)
                self.mem1_learn_counter += 1
        else:
            
            if len(self.memory3) > BATCH_SIZE_2:
                experiences = self.memory3.sample()
                self.learn(experiences, GAMMA)
                self.mem3_learn_counter += 1
         
        alpha = min(alpha_end, alpha_rate*alpha)        
    
    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        
        #print(experiences)
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        q_target_np = Q_targets.detach().cpu().numpy()
        q_expected_np = Q_expected.detach().cpu().numpy()
        self.q = abs(q_target_np - q_expected_np)
        
        if(self.mem_counter2 % BATCH_SIZE_2 == 0):
            if(np.max(self.q) < 10):
                self.memory3.add(states, actions, rewards, next_states, dones)
            #print(len(self.memory3))

        self.mem_counter2 += 1
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lenmem3 = self.mem_counter2
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        

        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
      
    def popleft(self):
        """retutn the 1st experience."""
        return self.memory.popleft()  
    
    # def pop(self,i):
    #     """retutn the 1st experience."""
    #     print(self.memory.pop(i))
    #     return self.memory.pop(i) 
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Replay2(ReplayBuffer):
    
    #def __init__(self, action_size, buffer_size, batch_size, seed):
        #Replay2.__init__(self,  action_size, buffer_size, batch_size, seed)
    
    def add(self, state, action, reward, next_state, done,_):
        experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done","q"])
    
        """Add a new experience to memory."""
        e = experience(state, action, reward, next_state, done,_)
        self.memory.append(e)
    
    