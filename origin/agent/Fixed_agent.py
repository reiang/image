import random
class FixedAgent():
    def __init__(self, mode, max_depth):
        self.mode = mode
        self.max_depth = max_depth

    def choose_action(self, state):
        action = self.mode * random.randint(1, self.max_depth) + self.max_depth - 1
        return action, None
    
    def store_transition(self, state, action, reward, next_state):
        pass
                
    def learn(self, ):
        pass