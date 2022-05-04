import numpy as np
import matplotlib.pyplot as plt

class Warehouse():

    def __init__(self):
        self.GRID_DIM = np.array([6,7])

        self.agent_position = np.array([2,1])

        self.box_position = np.array([3,4])
        self.goal_position = np.array([1,3])

        self.wall_position = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[1,0],[1,6],[2,0],[2,6],[3,0],[3,1],[3,2],[3,5],[3,6],[4,0],[4,1],[4,2],[4,5],[4,6],[5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6]])
        self.box_stuck=np.array([[1,1],[2,1],[4,3],[4,4],[1,5],[2,5]])
        
        self.action_space=4
        self.state_space=(14,14)
        
        self.state_lookup={(1,1):0,(2,1):1,(1,2):2,(2,2):3,(1,3):4,(2,3):5,(3,3):6,(4,3):7,(1,4):8,(2,4):9,(3,4):10,(4,4):11,(1,5):12,(2,5):13}
        
        self.agent_state=self.state_lookup[tuple(self.agent_position)]
        self.box_state=self.state_lookup[tuple(self.box_position)]
        self.goal_state=self.state_lookup[tuple(self.goal_position)]
        
        self.history=[[self.agent_position,self.box_position]]
        self.prev_ep_history=[]


    def reset(self):
        self.agent_position = np.array([2,1])

        self.box_position = np.array([3,4])
        
        

    def step(self, action):

        action_lookup={0:"left",1:"up",2:"right",3:"down"}
        change_coordinates = {"left":np.array([-1, 0]),"right":np.array([1, 0]),"down":np.array([0, 1]),"up":np.array([0, -1])}

        reward=-1
        done=False

        # Moving Agent
        new_agent_pos=np.sum([self.agent_position,change_coordinates[action_lookup[action]]],axis=0)
        new_box_pos=self.box_position

        # If agent in same position as wall
        if new_agent_pos.tolist() in self.wall_position.tolist():
            new_agent_pos=self.agent_position

        # New agent in same place as box
        elif new_agent_pos.tolist()==self.box_position.tolist():
            
            # Moving Box
            new_box_pos=np.sum([self.box_position,change_coordinates[action_lookup[action]]],axis=0)
            
            # If box in same position as wall
            if new_box_pos.tolist() in self.wall_position.tolist():
                new_box_pos=self.box_position
                new_agent_pos=self.agent_position
        
        self.agent_position=new_agent_pos
        self.box_position=new_box_pos
        
        self.agent_state=self.state_lookup[tuple(self.agent_position)]
        self.box_state=self.state_lookup[tuple(self.box_position)]

        # If bos is on goal
        if new_box_pos.tolist()==self.goal_position.tolist():
            reward=0
            done=True
            self.reset()
        if new_box_pos.tolist() in self.box_stuck.tolist():
            reward=-5
            done=True
            self.reset()
            

        return [[self.agent_state,self.box_state],reward,done]
       
        
    def render(self,for_animation=False): 
        
        img=np.ones((self.GRID_DIM[1],self.GRID_DIM[0],3),dtype=np.uint8)
        img=img*255

        for i in  self.wall_position:
            for j in range(0,3):
                img[i[1],i[0],j]=0

        img[self.box_position[1] ,self.box_position[0] ]=[255,0,0]

        img[self.agent_position[1] , self.agent_position[0]  ]=[255,255,0]

        img[self.goal_position[1] ,self.goal_position[0]  ]=[0,255,0]

        if not for_animation:
            plt.imshow(img)
            plt.plot(0,0,"-",color="red",label="Box")
            plt.plot(0,0,"-",color="green",label="Goal")
            plt.plot(0,0,"-",color="yellow",label="Agent")
            plt.legend(loc="upper right",bbox_to_anchor=(1.25,1))
            plt.show()
        else:
            return img
        