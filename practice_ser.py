import numpy as np
from SER_interplay import SERmodel_multneuro_buf_c
import random
import torch
print(torch.cuda.is_available())

# ===== STEP 1: make a random sparse network =====

np.random.seed(4) #ensure that it is the same random start of numbers
#empty adjacency matrix: 10 nodes, no links yet (rowa = from, cols = to)
print(np.zeros((10, 10)))
A = np.zeros((10, 10), dtype = int)
print(A) #check all should be 0s

# list every allowed (from, to) position for a link - exclude self-loops
spots = []
for row in range(10): #every "from" node
    for col in range(10): #every "to" node
        if row != col: #skip the diagonal (node -> itself)
            spots.append((row, col))
print(len(spots)) # check: should be 90

random.shuffle(spots) #mixes spots in a random order

#adds the exciting (+1) links to the netwok and checks it 
#first 10 random spots -> exciting links
for row, col in spots[:10]:
    A[row][col] = 1
#next 10 spots -> blocking links
for row, col in spots[10:20]:
    A[row, col] = -1

print(A) 
print((A == 1).sum()) #should be 10
print((A == -1).sum()) #should be 10
print(A.diagonal()) #should be 0s

#at this point have a random sparse network with 10 nodes, 10 exciting links and 10 blocking links - the wiring

# ==== STEP 2: the SER rules, tested
# used to check rules
T = np.zeros((3, 3), dtype=int)
T[0, 1] = 1 # 0 excites 1
T[1, 2] = 1 # 1 escites 2
T[2, 0] = 1 # 2 excites 0 (closes the loop)

#starting rw: node0=E, node1=S, node 2= R
states = np.array([1, 0, -1])
print(states)

# blank next row -> everyone S (0) to begin with
# rule 2 (R becomes S) happens automatically
new_states = np.zeros(3, dtype=int)

#SER rules written plainly before making them a function

for node in range(3): # go through each node
    #rule 1: E -> R
    if states[node] == 1:
        new_states[node] = -1
    #rule 3: S -> E if total push from firing nodes > 0
    if states[node] == 0: #current state must be S
        push = 0
        for other in range(3):
            if states[other] == 1: #only firing nodes push
                push = push + T[other, node] # arrow from other into node
                if push>0: #decide after adding everyone
                    new_states[node]=1
print(new_states)  # expect [-1 1 0] = R E S
# all just rules not function

#T=np.zeros((10,10), dtype=int)
#for i in range(10):
  #  for j in range(10):
   #     T[i, j] = random.randint(-1, 1)
T = np.random.randint(-1, 2, size=(10,10)) # random dense test network every box -1, 0, or 1 - only for testng ser_steps

# ===== CHECK: does my ser_steps match he repo's SERmodel? ====

# --- test network: a dense 10-nde matrix (every box -1, 0 or 1) ---
# (typed out so it's the same every run, rows = FROM, cols = T0)
# T = np.array([[ 1,  0, -1,  1, -1,  0,  1,  0, -1,  1],
#        [-1,  1, -1, -1,  0,  1,  1, -1,  0,  1],
#        [ 1, -1,  1, -1,  1,  1,  0,  1,  1,  1],
#        [ 0, -1, -1,  1,  1,  1,  0, -1,  0,  1],
#        [ 0, -1, -1,  1,  1, -1,  1,  1,  1, -1],
#        [ 1, -1,  0,  1,  0, -1,  1,  0,  1,  0],
#        [ 0,  0, -1,  1,  0,  1,  1, -1, -1,  1],
#        [ 1,  1,  1,  0,  0,  0, -1,  1,  0,  0],
#        [-1,  1, -1,  1,  0,  1,  1,  1,  0, -1],
#        [-1,  1,  1, -1, -1,  0, -1,  0, -1, -1]])
states = np.array([1, 0, -1])
print(states)

# ===== STEP 2b: my SER model as a function =====
def ser_steps(states, W):
    """One SER time step.
    In: states = curret state of each node (1=E, 0=S, -1=R)
        W      = wiring mareiz, W[from, to] (+1 excite, -1 block, 0 none)
    Out: new_states = the states one step later"""
    # next row starts as all S (0) -> rule 2 (R -> S) happens automatically
    new_states = np.zeros(len(states), dtype = int)

    for node in range(len(states)): #go through every node
        #RULE 1: E->R
        if states[node] == 1:
            new_states[node] = -1
        #RULE 3: S -> E if total push from firing nodes > 0
        if states[node] ==0:
            push = 0 #running total
            for other in range(len(states)): #check every other node
                if states[other]==1: # only firing nodes push
                    push = push + W[other, node] #arrow from other into node
            if push>0: # decide after adding everyone up
                    new_states[node]=1
    return new_states # lines up with "for node" - NOT inside the loop
#print(ser_steps(np.array([1, 0, -1]), T))

# --- a random starting row: one state per node (-1=R, 0=S, 1=E) ---
start = np.random.randint(-1, 2, size=10)
print(start)

# --- my model: run 9 steps, keep every row ---
row = start
my_rows= [row]
for step in range(9):
    row = ser_steps(row, T) # feed ub tge katest row each time
    my_rows.append(row)
my_table = np.array(my_rows) # 10 rows: the start + 9 steps

#repo version: same network, same start
out = SERmodel_multneuro_buf_c(np.array([T.T]), 10, [start])
repo_table = out.cpu().numpy()[0, 0].T


#print(my_table)
#print(repo_table)
#print(np.array_equal(my_table, repo_table))

# --- compare---
print("start:")
print(start) # check there are definitely some 1s, otherwise test is too easy and you will just get TRUE
print("mine, first 3 rows:")
print(my_table[:3])
#print(repo_table[:3])
print("repo, first 3 rows:") 
print(repo_table[:3])
print("shapes:", my_table.shape, repo_table.shape)
print(np.array_equal(my_table, repo_table)) # True = mu model matches the repo's
# RESULT: TUE for seeds 2, 3, 4