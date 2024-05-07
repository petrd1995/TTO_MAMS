import numpy as np
class Bar:
    def __init__(self, node1, node2, material, area, length, num=None):
        self.node1 = node1
        self.node2 = node2
        self.num = num
        self.material = material
        self.area = area
        self.length = length
        self.k_global = None
        self.pipe = False
        self.node_in = None
        self.node_out = None
        self.Tf_in = None
        self.r_inner = None
        self.r_outer = None
        
    @property
    def direction_cosines(self):

        L = self.length
        dx = self.node2.x - self.node1.x
        dy = self.node2.y - self.node1.y
        dz = self.node2.z - self.node1.z
        
        c1 = dx/L
        c2 = dy/L
        c3 = dz/L

        self.dc = np.array([c1, c2, c3])
        self.dc = np.reshape(self.dc, (3, 1))

        return self.dc

    def local_geometry_matrix(self):
        L = self.length
        c1, c2, c3 = self.direction_cosines.flatten()

        B = np.zeros((2, 6))
        B[0, :3] = [c1, c2, c3]   
        B[1, 3:] = [c1, c2, c3]   

        return B

    def truss_stiffness_matrix(self, transform=False):
        E = self.material.E
        A = self.area
        L = self.length

        if self.direction_cosines is None:
            dx = self.node2.x - self.node1.x
            dy = self.node2.y - self.node1.y
            dz = self.node2.z - self.node1.z
            
            c1 = dx/L
            c2 = dy/L
            c3 = dz/L

            self.direction_cosines = np.array([c1, c2, c3])
            self.direction_cosines = np.reshape(self.direction_cosines, (3, 1))

        c1 = self.direction_cosines[0]
        c2 = self.direction_cosines[1]
        c3 = self.direction_cosines[2]

        if isinstance(A,(float,int)):
            self.k_global = A*E/L*np.array([[c1**2, c1*c2, c1*c3, -c1**2, -c1*c2, -c1*c3],
                                            [c1*c2, c2**2, c2*c3, -c1*c2, -c2**2, -c2*c3],
                                            [c1*c3, c2*c3, c3**2, -c1*c3, -c2*c3, -c3**2],
                                            [-c1**2, -c1*c2, -c1*c3, c1**2, c1*c2, c1*c3],
                                            [-c1*c2, -c2**2, -c2*c3, c1*c2, c2**2, c2*c3],
                                            [-c1*c3, -c2*c3, -c3**2, c1*c3, c2*c3, c3**2]])
        else:
            self.k_global = E/L*np.array([[c1**2, c1*c2, c1*c3, -c1**2, -c1*c2, -c1*c3],
                                            [c1*c2, c2**2, c2*c3, -c1*c2, -c2**2, -c2*c3],
                                            [c1*c3, c2*c3, c3**2, -c1*c3, -c2*c3, -c3**2],
                                            [-c1**2, -c1*c2, -c1*c3, c1**2, c1*c2, c1*c3],
                                            [-c1*c2, -c2**2, -c2*c3, c1*c2, c2**2, c2*c3],
                                            [-c1*c3, -c2*c3, -c3**2, c1*c3, c2*c3, c3**2]])

        if transform:
            self.k_global = self.transform(self.k_global)

    def transform(self, matrix, angle, axis):

        T = np.zeros((6,6))
        
        if axis == 'x':
            T = np.array([[1, 0, 0, 0, 0, 0],
                          [0, np.cos(angle), np.sin(angle), 0, 0, 0],
                          [0, -np.sin(angle), np.cos(angle), 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, np.cos(angle), np.sin(angle)],
                          [0, 0, 0, 0, -np.sin(angle), np.cos(angle)]])

        elif axis == 'y':
            T = np.array([[np.cos(angle), 0, -np.sin(angle), 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [np.sin(angle), 0, np.cos(angle), 0, 0, 0],
                          [0, 0, 0, np.cos(angle), 0, -np.sin(angle)],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, np.sin(angle), 0, np.cos(angle)]])

        elif axis == 'z':
            T = np.array([[np.cos(angle), np.sin(angle), 0, 0, 0, 0],
                          [-np.sin(angle), np.cos(angle), 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, np.cos(angle), np.sin(angle), 0],
                          [0, 0, 0, -np.sin(angle), np.cos(angle), 0],
                          [0, 0, 0, 0, 0, 1]])
                          
        return T.T @ matrix @ T

    def truss_lumped_mass_matrix(self):
        if isinstance(self.area,(float,int)):
            self.mass = self.material.density * self.area * self.length
        else:
            self.mass = self.material.density * self.length
        self.mass_matrix_global = self.mass/2*np.eye(6)

    def truss_consistent_mass_matrix(self):
        if isinstance(self.area,(float,int)):
            self.mass = self.material.density * self.area * self.length
        else:
            self.mass = self.material.density * self.length
        self.mass_matrix_global = self.mass/6*np.array([[2, 0, 0, 1, 0, 0],
                                                        [0, 2, 0, 0, 1, 0],
                                                        [0, 0, 2, 0, 0, 1],
                                                        [1, 0, 0, 2, 0, 0],
                                                        [0, 1, 0, 0, 2, 0],
                                                        [0, 0, 1, 0, 0, 2]])

    @staticmethod
    def radius_to_area(radius):
        return np.pi * radius ** 2
        
    @staticmethod
    def area_to_radius(area):
        return np.sqrt(area / np.pi)