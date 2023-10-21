class Memory:

    def __init__(self):
        self.nodes = []
        self.edge_attributes = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def clear(self):
        self.nodes.clear()
        self.edge_attributes.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
