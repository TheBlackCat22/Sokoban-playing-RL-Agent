import Warehouse_Env
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def e_greedy(state,Q,epsilon):
    n=np.random.random_sample()
    if n>=epsilon:
        action=np.argmax(Q[state[0],state[1]])
    else:
        action=np.random.randint(0,4)
    return action
    
        
            
def sarsa(env,alpha=0.5,gamma=0.1,epsilon=0.1,episodes=100,verbose=0):          
    
    Q=np.random.random_sample(size=(env.state_space[0],env.state_space[1],env.action_space))
    Q[:,env.goal_state,:] = 0.0
    
    policy=np.zeros(shape=(env.state_space[0],env.state_space[1]))
    
    for ep in range(1,episodes+1):
        steps=0
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
                if verbose==1:
                    print(f"Episode {ep} completed with {steps} steps")
                
    for i in range(env.state_space[0]):
        for j in range(env.state_space[1]):
            policy[i,j]=np.argmax(Q[i,j])
            
    return policy
          
                
                
def q_learning(env,alpha=0.5,gamma=0.1,epsilon=0.1,episodes=100,verbose=0):
    
    Q=np.random.random_sample(size=(env.state_space[0],env.state_space[1],env.action_space))
    Q[:,env.goal_state,:] = 0.0
    
    policy=np.zeros(shape=(env.state_space[0],env.state_space[1]))
    
    for ep in range(1,episodes+1):
        steps=0
        state=[env.agent_state,env.box_state]
        done=False
        while done==False:
            #print(state,action,done)
            action=e_greedy(state,Q,epsilon)
            [new_state,reward,done]=env.step(action)
            steps+=1
            Q[state[0],state[1],action]=Q[state[0],state[1],action] + alpha*(reward+(gamma*np.max(Q[new_state[0],new_state[1]]))-Q[state[0],state[1],action])
            state=new_state
            if done==True:
                 if verbose==1:
                    print(f"Episode {ep} completed with {steps} steps")
                
    for i in range(env.state_space[0]):
        for j in range(env.state_space[1]):
            policy[i,j]=np.argmax(Q[i,j])
            
    return policy
    
    

def on_policy_mc(env,gamma=0.1,epsilon=0.1,episodes=100,verbose=0):
    
    Q=np.random.random_sample(size=(env.state_space[0],env.state_space[1],env.action_space))
    returns=np.zeros(shape=Q.shape+(episodes,))
    
    policy=np.zeros(shape=(env.state_space[0],env.state_space[1]))
    
    for ep in range(1,episodes+1):
        done=False
        history=[]
        while not done:
            state=[env.agent_state,env.box_state]
            action=e_greedy(state,Q,epsilon)
            [new_state,reward,done]=env.step(action)
            history.append([state,action,reward])
            steps=len(history)
            state=new_state
        g=0
        for t in range(len(history)-1,-1,-1):
            g = (gamma*g) + history[t][2]
            if history[t][0:2] not in np.array(history,dtype=object)[:t,0:2].tolist():
                returns[history[t][0][0],history[t][0][1],history[t][1],ep-1]=g
                Q[history[t][0][0],history[t][0][1],history[t][1]]=np.average(np.array(returns)[history[t][0][0],history[t][0][1],history[t][1],0:ep])   
        if verbose==1:
            print(f"Episode {ep} completed with {steps} steps")
    
    for i in range(env.state_space[0]):
        for j in range(env.state_space[1]):
            
            n=np.random.random_sample()
            if n>=epsilon:
                action=np.argmax(Q[state[0],state[1]])
                policy[i,j]=action
            else:
                action=np.random.randint(0,4)
                policy[i,j]=action
    
    return policy
    
    

def off_policy_mc(env,gamma=0.1,episodes=100,verbose=0):
    
    def soft_policy():
        return np.random.randint(0,4)
    
    Q=np.random.random_sample(size=(env.state_space[0],env.state_space[1],env.action_space))
    C=np.zeros_like(Q)
    
    policy=np.zeros(shape=(env.state_space[0],env.state_space[1]))
    
    for i in range(env.state_space[0]):
        for j in range(env.state_space[1]):
            policy[i,j]=np.argmax(Q[i,j])
    
    for ep in range(1,episodes+1):
        done=False
        history=[]
        while not done:
            state=[env.agent_state,env.box_state]
            action=soft_policy()
            [new_state,reward,done]=env.step(action)
            history.append([state,action,reward])
            steps=len(history)
            state=new_state
        g=0
        w=1
        for t in range(len(history)-1,-1,-1):
            g = (gamma*g) + history[t][2]
            C[history[t][0][0],history[t][0][1],history[t][1]]=C[history[t][0][0],history[t][0][1],history[t][1]]+w
            Q[history[t][0][0],history[t][0][1],history[t][1]]=Q[history[t][0][0],history[t][0][1],history[t][1]] + (w/ C[history[t][0][0],history[t][0][1],history[t][1]])*(g-Q[history[t][0][0],history[t][0][1],history[t][1]])
            policy[history[t][0][0],history[t][0][1]]=np.argmax(Q[history[t][0][0],history[t][0][1]])
            if policy[history[t][0][0],history[t][0][1]] == history[t][1]:
                w=w*4
        if verbose==1:
            print(f"Episode {ep} completed with {steps} steps")
    return policy   
   
   
                
def play_policy(env,policy):
    state=[env.agent_state,env.box_state]
    done=False
    step=0
    
    frame=[]
    frame.append([plt.imshow(env.render(for_animation=True),animated=True)])
    
    fig=plt.figure()
    
    while not done:
        step+=1
        action=policy[state[0],state[1]]
        [new_state,_,done]=env.step(action)
        frame.append([plt.imshow(env.render(for_animation=True),animated=True)])
        state=new_state
        if step==50:
            break
    
    anim = animation.ArtistAnimation(fig, frame,blit=True) 
    return anim