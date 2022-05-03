import numpy as np
import Warehouse_Env

def e_greedy(state,Q,epsilon):
    n=np.random.random_sample()
    if n<=epsilon:
        action=np.argmax(Q[state[0],state[1]])
    else:
        action=np.random.randint(0,4)
    return action
        
            
def sarsa(env,alpha=0.5,gamma=0.1,epsilon=0.25,episodes=10):          
    
    Q=np.ones(shape=(env.state_space[0],env.state_space[1],env.action_space),dtype=np.float16)
    Q[:,env.goal_state,:] = 0.0
    
    steps=0
    for ep in range(1,episodes+1):
        state=[env.agent_state,env.box_state]
        action=e_greedy(state,Q,epsilon)
        done=False
        while done==False:
            #print(state,action,done)
            [new_state,reward,done]=env.step(action)
            steps+=1
            new_action=e_greedy(new_state,Q,epsilon)
            Q[state[0],state[1],action]=Q[state[0],state[1],action] + alpha*(reward+(gamma*Q[new_state[0],new_state[1],new_action])-Q[state[0],state[1],action])
            state=new_state
            action=new_action
            if done==True:
                print(f"Episode {ep} completed on step {steps}")
                
env=Warehouse_Env.Warehouse()
sarsa(env)
            

            

        