import numpy as np
from collections import defaultdict

#Monte Carlo Agent with exploring starts
class MonteCarloAgent_ES:
    def __init__(self):
        self.action_space = [-1, 0, 1]
        # might needs to be replaced with random values for Q that sum to one
        self.Q = defaultdict(zero_array)
        self.N = defaultdict(zero_array)
        self.returns = defaultdict(zero_array)
        self.epsilon = 1.0
        self.gamma = 0.9  # discount factor

    def update_Q(self, episode, total_reward):
        G=0
        #print("updateQ")
        #print(episode) 
        sa= ()
        state_checklist=[]
        for s, a, r in episode:
            state_checklist.append((s,a))
        i=0
        k=0
        for s, a, r in reversed(episode):
            state_checklist.pop()
            i+=1
            G= G*self.gamma + r
            # print(s, a, r) s[0]= ballposition (a,b), s[1] =ball direction tuple, s[2] is paddle postion tuple, s[3] is paddle speed, s[4] is bricks in iteratable
            sa = (tuple(s[0]), tuple(s[1]), tuple(s[2]),s[3], tuple(tuple(tuple(brick) for brick in bricks) for bricks in s[4]), a)
            if (s,a) in state_checklist:
                k+=1
                #print('i: {} ;k/i :{}'.format(i,k/i))
            else:
                self.N[sa] += 1
                #self.Q[sa] += (total_reward - self.Q[sa]) / self.N[sa]
                self.Q[sa] = (self.Q[sa]*(self.N[sa]-1)+ G)/self.N[sa]

            

    def act(self, state, train=True):
        #if train and np.random.rand() <= self.epsilon:
        #    return np.random.choice(self.action_space)
        #else:
        state_action_tuple=()
        q_values=[]
        # make sure that a random action is taken that has the max value
        
        for action in self.action_space:
            state_action_tuple =(tuple(state[0]), tuple(state[1]), tuple(state[2]), state[3],tuple(tuple(tuple(brick) for brick in bricks) for bricks in state[4]), action)
            q_values.append(self.Q[state_action_tuple])

        # randomly (argmax alone takes allways the first option)
        max_q_value = np.max(q_values)
        max_q_indices = [i for i, q_value in enumerate(q_values) if q_value == max_q_value]

        return self.action_space[np.random.choice(max_q_indices)]

        #return self.action_space[np.argmax(q_values)]

    def update_epsilon(self, t):
        self.epsilon = max(1.0 / (1.0 + t), 0.05)

#################################################################################################
# Monte carlo first visit 
class MonteCarloAgent_FV:
    def __init__(self):
        self.action_space = [-1, 0, 1]
        # might needs to be replaced with random values for Q that sum to one
        self.Q = defaultdict(zero_array)
        self.N = defaultdict(zero_array)
        self.returns = defaultdict(zero_array)
        self.epsilon = 1.0
        self.gamma = 0.9  # discount factor

    def update_Q(self, episode, total_reward):
        G=0
        #print("updateQ")
        #print(episode) 
        sa= ()
        state_checklist=[]
        for s, a, r in episode:
            state_checklist.append((s,a))
        i=0
        k=0
        for s, a, r in reversed(episode):
            state_checklist.pop()
            i+=1
            G= G*self.gamma + r
            # print(s, a, r) s[0]= ballposition (a,b), s[1] =ball direction tuple, s[2] is paddle postion tuple, s[3] is paddle speed, s[4] is bricks in iteratable
            sa = (tuple(s[0]), tuple(s[1]), tuple(s[2]),s[3], tuple(tuple(tuple(brick) for brick in bricks) for bricks in s[4]), a)
            if (s,a) in state_checklist:
                k+=1
                #print('i: {} ;k/i :{}'.format(i,k/i))
            else:
                self.N[sa] += 1
                #self.Q[sa] += (total_reward - self.Q[sa]) / self.N[sa]
                self.Q[sa] = (self.Q[sa]*(self.N[sa]-1)+ G)/self.N[sa]

            

    def act(self, state, train=True):
        if train and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            state_action_tuple = ()
            q_values = []
            for action in self.action_space:
                state_action_tuple = (tuple(state[0]), tuple(state[1]), tuple(state[2]), state[3],
                                    tuple(tuple(tuple(brick) for brick in bricks) for bricks in state[4]), action)
                q_values.append(self.Q[state_action_tuple])

            max_q_value = np.max(q_values)
            max_q_indices = [i for i, q_value in enumerate(q_values) if q_value == max_q_value]

            return self.action_space[np.random.choice(max_q_indices)]


    def update_epsilon(self, t):
        self.epsilon = max(1.0 / (1.0 + t), 0.05)


#################################################################################################

# function return function (as needed by defaultdict)
# to create array with rand values that  sum to 1
def normalizedProps(size):
    randomArray=np.random.rand(size)
    randomArray/= randomArray.sum()
    return (lambda: randomArray)

#otherwise pickle does not work
def zero_array():
    return np.zeros(1)