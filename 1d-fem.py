import numpy as np
from matplotlib import pyplot as plt

def print_matrix(matrix, name):
    """Prints the matrix with the given name."""
    print(f"{name} = [")
    for row in matrix:
        print("  ", row)
    print("]\n")

#Material Property Definition Step

E = 100



# Example local stiffness matrices for 2 elements (simple 1D case for illustration)
# Each element has 2 nodes, each with 2 DOFs, hence Ke is 4x4
Ke1 = np.array([
    [12, -6, 0, 0],
    [-6, 12, 0, -6],
    [0, 0, 12, -6],
    [0, -6, -6, 12]
])

Ke2 = np.array([
    [12, -6, 0, 0],
    [-6, 12, 0, -6],
    [0, 0, 12, -6],
    [0, -6, -6, 12]
])

# Assuming node connectivity: [0, 1] for element 1, [1, 2] for element 2
conn1 = [0, 1]
conn2 = [1, 2]

print("Connectivities")
print(conn1)
print(conn2)

# Initialize global stiffness matrix (3 nodes * 2 DOFs each = 6x6)
num_nodes = 3
num_dofs = num_nodes * 2
K = np.zeros((num_dofs, num_dofs))

print_matrix(K, "empty stiffness matrix")

# Function to assemble the local stiffness matrix into the global stiffness matrix
def assemble_global_stiffness(K, Ke, conn):
    for i, I in enumerate(conn):
        for j, J in enumerate(conn):
            K[2*I, 2*J] += Ke[2*i, 2*j]
            K[2*I+1, 2*J] += Ke[2*i+1, 2*j]
            K[2*I+1, 2*J+1] += Ke[2*i+1, 2*j+1]
            K[2*I, 2*J+1] += Ke[2*i, 2*j+1]
    return K

# Assemble global stiffness matrix with detailed steps
print("Initial Global Stiffness Matrix:")
print_matrix(K, "K")

print("Assembling Element 1:")
print_matrix(Ke1, "Ke1")
K = assemble_global_stiffness(K, Ke1, conn1)
print_matrix(K, "K")

print("Assembling Element 2:")
print_matrix(Ke2, "Ke2")
K = assemble_global_stiffness(K, Ke2, conn2)
print_matrix(K, "K")

print("Final Global Stiffness Matrix:")
print_matrix(K, "K")

f = np.zeros(2*num_nodes)
print_matrix(f, "initial force matrix")


nodes = [0 , 1, 2]

fixed_node = 0

for i in [2*fixed_node, 2*fixed_node+1]:
    K[i, :] = 0.0 #Row to 0
    K[:, i] = 0.0 #Column to 0
    K[i, i] = 1.0 #Diagonals to 1.0

print_matrix(K, "updated global stiffness matrix with fixed initial element")

f[2*2+1] = -1

print_matrix(f, "force matrix")

u = np.linalg.solve(K,f)
    
# Reshape displacements for easier interpretation
ux = u[0::2]  # x-displacements
uy = u[1::2]  # y-displacements

# Initial nodal positions for plotting
nodal_positions = np.array([[0, 0], [1, 0], [2, 0]])

# Deformed nodal positions
deformed_positions = nodal_positions + np.column_stack((ux, uy))

# Plot the initial and deformed mesh
plt.figure(figsize=(8, 4))
plt.plot(nodal_positions[:, 0], nodal_positions[:, 1], 'bo-', label='Initial')
plt.plot(deformed_positions[:, 0], deformed_positions[:, 1], 'ro--', label='Deformed')


plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Initial and Deformed Mesh with Applied Forces')
plt.grid()
plt.axis('equal')
plt.show()