import numpy as np
import time
import cv2 as cv
import heapq
from tqdm import tqdm
import math


nodes={}                        # To store all nodes
start_node=None                 # To store the start node
end_node=None                   # To store the end node

step_size=20                    # Step size for the robot
animation=False                 # To display the animation of the path planning


# Visualizing the path by writing video
frameSize = (600, 250)
fourcc = cv.VideoWriter_fourcc('m','p','4','v')
out  = cv.VideoWriter('output.mp4', fourcc, 250, frameSize)

# Function to create the map and obstacle map
def create_map():
    # Main map for display
    map=np.ones((250,600,3),dtype=np.uint8)
    map[:,:,0]=51
    map[:,:,1]=51
    map[:,:,2]=44

    # Obstacle map for path planning
    obs_map=np.ones((250,600),dtype=np.uint8)*255

    # Draw obstacles on the maps with clearance
    for i in range(600):
        for j in range(250):
            # Walls
            if i<10 or i>590 or j<10 or j>240:
                map[j][i]=(222,228,203)
                obs_map[j][i]=0
            # Rectangles
            if ((j>=10 and j<=110) or (j>=140 and j<=240)) and (i>=90 and i<=160):
                map[j][i]=(222,228,203)
                obs_map[j][i]=0
            # Hexagon
            if (85/2) * abs(i-300)/85 + 40 <= j <= 250 - (85/2) * abs(i-300)/85 - 40 and 225 <= i <= 375:
                map[j][i]=(222,228,203)
                obs_map[j][i]=0
                
            # Sides of the triangle for drawing clearance
            if i==460 and j>=25 and j<=225:
                map[j][i]=(222,228,203)
                obs_map[j][i]=128
            if j-2*i+895==0 and 460<=i<=510:
                map[j][i]=(222,228,203)
                obs_map[j][i]=128
            if j+2*i-1145==0 and 460<=i<=510:
                map[j][i]=(222,228,203)
                obs_map[j][i]=128
            
    # Draw obstacles on the maps          
    for i in range(600):
        for j in range(250):
            # Rectangles without clearance
            if ((j>=0 and j<=100) or (j>=150 and j<=250)) and (i>=100 and i<=150):
                map[j][i]=(136, 131, 14)
            # Triangle without clearance
            if (200/100) * (i-460) + 25 < j < (-200/100) * (i-460) + 225 and 460 < i < 510:
                map[j][i]=(136, 131, 14)
                obs_map[j][i]=0
            # Hexagon without clearance
            if (75/2) * abs(i-300)/75 + 50 <= j <= 250 - (75/2) * abs(i-300)/75 - 50 and 235 <= i <= 365:
                map[j][i]=(136, 131, 14)
    
    # Draw clearance on the maps for triangle
    for i in range(460,511):
        for j in range(25,226):
            if obs_map[j][i]==128:
                for k in range(-10,11):
                    for l in range(-10,11):
                        if obs_map[j+k][i+l]!=128 and map[j+k][i+l][0]!=136:
                            obs_map[j+k][i+l]=0
                            map[j+k][i+l]=(222,228,203)

    
    # cv.imshow('Map',map)
    # cv.waitKey(0)   
    return map,obs_map

# Function to insert a node into the nodes dictionary
def insert_node(cost=None,node=None,parent=None):    
    if len(nodes)==0:
        # Inserting obstacle nodes into the nodes dictionary
        for i in range(600):
            for j in range(250):
                if obs_map[j][i]!=255:
                    nodes.update({(i,j,0):[None,float('inf')]})
    else:
        nodes.update({node:[parent,cost]})
        
# Action functions to move the mobile robot in 5 directions
def Actions(node,step_size):
    def action(node,step_size,angle):
        i,j,th=node
        theta=np.deg2rad(th+angle)
        i=int(i+step_size*np.cos(theta))
        j=int(j+step_size*np.sin(theta))
        th=(th+angle)%360
        cost=step_size
        if check_if_duplicate((i,j,th)):
            return None,None
        else:
            return (i,j,th),cost
    
    def check_if_duplicate(node):
        x,y,th=node
        for i in range(-1,2):
            for j in range(-1,2):  
                for k in range(-6,7):              
                    if (x+i*0.5,y+j*0.5,th+k*30) in nodes.keys():
                        return True
        return False
    
    return [action(node,step_size,-60),action(node,step_size,-30),action(node,step_size,0),action(node,step_size,30),action(node,step_size,60)]

# Function to calculate the total cost of a node f(n)=h(n)+g(n): h(n)=euclidean heuristic from node to end node, g(n)=cost of node   
def total_cost(node):
    i,j,_=node
    return np.round(math.sqrt((i-end_node[0])**2+(j-end_node[1])**2),2) +nodes[node][1]

# Function to check if the node is the end node
def check_if_end(node):
    if math.sqrt((node[0] - end_node[0])**2 + (node[1] - end_node[1])**2) <= 1.5:
        if node[2]==end_node[2]:
            return True
    else:
        return False

# Tree function to generate the heap tree graph of the nodes using A* Algorithm
def tree():
    global nodes
    open_list=[]
    closed_list=set()                  
    tot_cost=total_cost(start_node)    
    heapq.heappush(open_list,(tot_cost,start_node))
    return_node=None
    
    # For animation of progress bar
    print('\nSearching for path:')
    for i in tqdm(range(100)):
        while open_list:
            _,current_node=heapq.heappop(open_list)
            current_c2c=nodes[current_node][1]      
            closed_list.add(current_node)                   

            if current_node!=start_node:
                cv.arrowedLine(map,(nodes[current_node][0][0],250-nodes[current_node][0][1]),(current_node[0],250-current_node[1]),(0,0,0),1)
                cv.waitKey(1)
                out.write(map)
            
            if animation:
                cv.imshow('Map',map)
                # cv.waitKey(1)
            
            if check_if_end(current_node):
                open_list=None
                return_node=current_node
                break              
            
            for action in Actions(current_node,step_size):
                new_node,cost=action
                if new_node is not None and new_node not in closed_list and (new_node[0] in range(0,600)) and (new_node[1] in range(0,250)):
                    new_cost=current_c2c+cost
                    
                    # Checking if node is already explored and if the new cost is less than the previous cost
                    if new_node not in nodes.keys() and ((new_node[0],new_node[1],0) not in nodes.keys()) and obs_map[new_node[1]][new_node[0]]==255:
                        insert_node(new_cost,new_node,current_node)
                        new_total_cost=total_cost(new_node)
                        heapq.heappush(open_list,(new_total_cost,new_node))
                    else:
                        for i in range(len(open_list)):
                            if open_list[i][1]==new_node and open_list[i][0]>new_cost:
                                insert_node(new_cost,new_node,current_node)
                                new_total_cost=total_cost(new_node)
                                open_list[i]=(new_total_cost,new_node)
        time.sleep(0.01)
        i+=1
    return return_node
          
# Returns the parent node for a given node    
def get_parent(node):
    return nodes[node][0]

# Returns a path from the end_node to the start_node
def generate_path():
    # Searching starts here
    path=[tree()]
    if path[0] is not None:
        total_cost=nodes[path[0]][1]
        parent=get_parent(path[0])
        while parent is not None:
            path.append(parent)
            parent = get_parent(parent)
        path.reverse()
        print("Path found")
        return path,total_cost
    else:
        print("\nError: Path not found\nTry changing the orientation of the nodes")
        exit()

# Saving the map
def save_map(map):
    print("\nSaving the map:")
    for i,j in zip(range(len(path)-1),tqdm(range(len(path)-1))):
        if animation:
            cv.imshow('Map',map)
            cv.waitKey(1)
        cv.arrowedLine(map,(path[i][0],250-path[i][1]),(path[i+1][0],250-path[i+1][1]),(0,0,255),1)
        out.write(map)
    for i in range(500):
        out.write(map)
    print("Map saved as output.mp4\n")
    cv.waitKey(500)
    cv.destroyAllWindows()
    out.release()

# Getting user inputs
def get_inputs():
    global start_node,end_node,step_size,animation
    check_input=True
    while check_input:
        step=input("Enter the step size(1-10): ")
        step_size=int(step)
        if step_size>10 or step_size<1:
            print("Please enter a valid step size.")
        else:
            check_input=False
    
    animate=input("Do you want to visualize the path? (y/n): ")
    animation=(lambda x: True if x=='y'or x=='Y' else False)(animate)
    
    check_input=True
    while check_input:
        s_node=input("\nNote:'(10,10) is the starting point due to clearance on the walls and orientation should be multiple of 30 degrees'\nEnter the start node in the format 0 1 30 for (0,1,30): ")
        x,y,th=s_node.split()
        if int(x)>600 or int(y)>250 or int(x)<0 or int(y)<0 or int(th)%30!=0:
            print("Please enter valid coordinates.")
        elif (int(x),int(y),int(th)) in nodes.keys():
            print("Please enter a valid start node(Node in obstacle place).")
        else:
            check_input=False
            start_node=(int(x),int(y),int(th)%360)
            insert_node(0,start_node,None)

    check_input=True
    while check_input:
        f_node=input("\nNote:'(590,240) is the ending point due to clearance on the walls and orientation should be multiple of 30 degrees'\nEnter the end node in the format 0 1 30 for (0,1,30) : ")
        x,y,th=f_node.split()
        if int(x)>600 or int(y)>250 or int(x)<0 or int(y)<0 or int(th)%30!=0 or (int(x),int(y))==start_node:
            print("Please enter valid coordinates.")
        elif (int(x),int(y),int(th)) in nodes.keys():
            print("Please enter a valid end node(Node in obstacle place).")
        else:
            check_input=False
            end_node=(int(x),int(y),int(th)%360)

# Main function
if __name__ == "__main__":
    start_time = time.time()
    
    # Getting maps
    map,obs_map=create_map()
    
    # Inserting obstacle nodes into the nodes dictionary
    insert_node()

    # Getting user inputs
    get_inputs()
    
    # Generating the path and the total cost
    path,total_cost=generate_path()   
    
    # Saving the animation of the path generation
    save_map(map)
        
    end_time = time.time()
    print(f"Time taken to execute the code is: {(end_time-start_time)/60} minutes.")
    print("\nTotal cost of the path is: ",total_cost)
    print("\nPath is: ",path)
