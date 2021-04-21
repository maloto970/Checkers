class MCTS:

    def __init__(self, s0):
        self.s0 = s0

    def select(self, visits=[self.s0]):
        
        if len(self.s0.children) == 0:
            self.expand()
            self.backup()
            return
        
        for c in self.s0.children:


    # visit each selected node and update q value based on each subtrees   
    def backup(self):




class Node:
    def __init__(self, action, parent, game):
        self.action = action
        self.parent = parent
        self.children = None
        self.game = game

    def add_child(self, child):
        self.children.append(child)

    def update(self):

        if len(self.children) == 0:
            return 

        for c in self.children:
            if len(c.node.children) == 0:
                c.n += 1
                c.w += c.v
                c.q = c.w/c.n
                continue
            
            

class Edge:
    def __init__(self,p,v,node):
        self.node = node
        self.p = p
        self.v = v
        self.q = 0
        self.w = 0
        self.n = 0
        