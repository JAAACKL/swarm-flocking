# MEAM 6240, UPenn

class Edge2D:
    def __init__(self, in_nbr, out_nbr, cost=0):
        """Constructor"""
        self.in_nbr = in_nbr    # source node ID
        self.out_nbr = out_nbr  # destination node ID
        self.cost = cost        # edge cost (if needed)
    
    def __str__(self):
        """Printing"""
        return f"Edge ({self.in_nbr}, {self.out_nbr}), cost = {self.cost}" 