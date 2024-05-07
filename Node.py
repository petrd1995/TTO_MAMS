class Node:
    def __init__(self, x, y, z, num, forces=[0, 0, 0]):
        self.x = x
        self.y = y
        self.z = z
        self.coordinates = [x, y, z]
        self.num = num
        self.boundary_conditions = None 
        self.displacements = [0, 0, 0]
        self.forces = [0,0,0]
        self.thermal_load = None
        self.temperature = None
        self.bc_force_label = None
        self.bc_clamp_label = None
        self.bc_heat_label = None
        self.concentrated_mass = None

    def distance_to(self, other_node):
        dx = self.x - other_node.x
        dy = self.y - other_node.y
        dz = self.z - other_node.z
        return (dx**2 + dy**2 + dz**2)**0.5

    def set_displacements(self, u):
            self.displacements[0] = u[0]
            self.displacements[1] = u[1]
            self.displacements[2] = u[2]
            
    def __hash__(self):
        return hash(self.num)

    def __eq__(self, other):
        return self.num == other.num