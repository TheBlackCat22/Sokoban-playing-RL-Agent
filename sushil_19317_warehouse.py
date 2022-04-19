import numpy as np
import matplotlib.pyplot as plt

class WarehouseAgent():

    def __init__(self):
        self.GRID_DIM = np.array([6,7])

        self.agent_position = np.array([2,1])

        self.box_position = np.array([3,4])
        self.goal_position = np.array([1,3])

        self.wall_position = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[1,0],[1,6],[2,0],[2,6],[3,0],[3,1],[3,2],[3,5],[3,6],[4,0],[4,1],[4,2],[4,5],[4,6],[5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6]])


    def reset(self):
        self.agent_position = np.array([2,1])

        self.box_position = np.array([3,4])

        print(f"Agent Location = {self.agent_position}")
        print(f"Box Location = {self.box_position}")

        return [self.agent_position,self.box_position]


    def step(self, action):

        action_lookup={"a":"left","w":"up","d":"right","s":"down"}
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

        # If bos is on goal
        if new_box_pos.tolist()==self.goal_position.tolist():
            reward=0
            done=True
            self.reset()

        return [self.agent_position,reward,done]
        
    def render(self): 
        
        img=np.ones((self.GRID_DIM[1],self.GRID_DIM[0],3))
        img=img*255

        for i in  self.wall_position:
            for j in range(0,3):
                img[i[1],i[0],j]=0

        img[self.box_position[1] ,self.box_position[0]  ,0]=255
        img[self.box_position[1] ,self.box_position[0]  ,1]=0
        img[self.box_position[1] ,self.box_position[0]  ,2]=0

        img[self.agent_position[1] , self.agent_position[0]  ,0]=255
        img[self.agent_position[1] , self.agent_position[0]  ,1]=255
        img[self.agent_position[1] , self.agent_position[0]  ,2]=0

        img[self.goal_position[1] ,self.goal_position[0]  ,0]=0
        img[self.goal_position[1] ,self.goal_position[0]  ,1]=255
        img[self.goal_position[1] ,self.goal_position[0]  ,2]=0

        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.plot(0,0,"-",color="red",label="Box")
        plt.plot(0,0,"-",color="green",label="Goal")
        plt.plot(0,0,"-",color="yellow",label="Agent")
        plt.legend(loc="upper right",bbox_to_anchor=(1.25,1))
        plt.show()