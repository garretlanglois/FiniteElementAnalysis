import numpy as np
from matplotlib import pyplot as plt
import math

def shape(xi):
    #Defines a tuple that has the size xi for x_i elements
    x,y = tuple(xi)

    #Define the shape functions
    N = [(1.0-x)*(1.0-y), (1.0+x)*(1.0-y), (1.0+x)*(1.0+y), (1.0 - x)*(1.0+y)]
    return 0.25*np.array(N)

#Now we need the derivative of the shape functions as this will become what we need for our Jacobian
#Therefore we will define the following function

def gradshape(xi):
    x,y = tuple(xi)

    #This will be an array composed of two different arrays the first will be partial with respect to x and the second will be the partial with respect to y
    dN = [
        [-(1.0-y), (1.0-y), (1.0+y), -(1.0+y)],
        [-(1.0-x), -(1.0+x), (1.0+x), (1.0-x)]
    ]
    
    #Remember that the shape functions always have a 1/4 in front of them as their coefficient
    return 0.25*np.array(dN)

#Now we need to create the mesh

print('CREATING MESH')

#Number of elements in the x direction
mesh_ex = 9

#Number of elements in the y direction
mesh_ey = 49

#Size of the mesh in the x and y directions (i.e. the size of the object)
mesh_lx = 30.0
mesh_ly = 80.0

#The number of nodes in each direction
mesh_nx = mesh_ex + 1
mesh_ny = mesh_ey + 1

#The total number of nodes and elements
num_nodes = mesh_nx * mesh_ny

num_elements = mesh_ex * mesh_ey

#Mesh element sizes
mesh_hx = mesh_lx / mesh_ex
mesh_hy = mesh_ly / mesh_ey

nodes = []

#Define the locations of all of our nodes in the undeformed configuration

for y in np.linspace(0.0, mesh_ly, mesh_ny):
    for x in np.linspace(0.0, mesh_lx, mesh_nx):
        nodes.append([x,y])

#Create an numpy array with the nodes
nodes = np.array(nodes)

#Define the connection of evey node in the mesh
conn = []

#Picks the node in the bottom let corner of the element and defines a connection between all of the neighbouring nodes
for j in range(mesh_ey):
    for i in range(mesh_ex):
        n0 = i + j*mesh_nx
        conn.append([n0, n0 + 1, n0 + 1 + mesh_nx, n0 + mesh_nx])

#Now the fun part, the material model

print('MATERIAL MODEL')

E = 100.0
v = 0.48

#Define the stiffness matrix
C = E/(1.0+v)/(1.0-2.0*v)*np.array([[1.0-v, v, 0.0],
                                    [v, 1.0-v, 0.0],
                                    [0.0, 0.0, 0.5-v]])

print("GLOBAL STIFFNESS MATRIX ")
#This is the pretty hard conceptual part in my opinion
#The idea beind this comes from the fact that a global stiffness matrix can be created to solve for the displacement matrix everywhere in the system.
#This follows the formula      sum(B^(T)CBJw)*U = sum(H^TbJw) + sum(H^TtJw) + f
# Where B is the Strain-Displacement Matrix given by
# | dN1/dx   0     dN2/dx   0     dN3/dx   0     dN4/dx   0   |
# |   0    dN1/dy   0    dN2/dy   0    dN3/dy   0    dN4/dy |
# | dN1/dy dN1/dx dN2/dy dN2/dx dN3/dy dN3/dx dN4/dy dN4/dx |

#This can be used to solve the equation [eps_11, eps_22, eps_12] = B^(m)U where U is the nodal displacement vector [u_1, v_1, u_2, v_2, u_3, v_3, u_4, v_4] for every node in the element.

#Initializes the Global Stiffness Matrix there is 2* here because there is 2 degrees of freedom, x and y

K = np.zeros((2*num_nodes, 2* num_nodes))

#Gaussian Quadrature Section
#The importance of Guassian Quadrature is that it allows us to convert the integrals to sum functions with very high accuracy in comparison to other methods such as Riemann sums.
#The idea comes from the fact that we need to choose the weights and nodes so that we can get a very exact result.
#The method works from -1 to 1 and allows us to convert the integral across this domain into a sum of the function at certain "nodes" as long as they are multiplied by weights
#These nodes are points that are placed along the x axis from -1 to 1. It allows us to find "exact" answers for nth degree polynomials where degree is < 2n - 1 where n is the number of nodes we use.
#For this solution we will use 2 points -> 2(2) - 1 > which means it will give exact solutions for degrees 1 and 2 which is what we require as we are taking at most Area integrals over our domain.
#This is because we are in 2 dimensions. The idea states that the integral of f is approx equal to sum of f(x_i)*w_i.
#Now we need our weights and our points. The points come from the roots of the Legrande polynomials which are orthonormal polynomials which form a basis set that spans all polynomials. 
#We now choose the third Legrande polynomial as it will provide a solution for all polynomials for it's nth degree. This polynomial is 1/2(3x^2 - 1) which has roots 1/sqrt(3).

q4 = [[x/math.sqrt(3.0),y/math.sqrt(3.0)] for y in [-1.0,1.0] for x in [-1.0,1.0]]

#All of these matrixes will need to be updated in the future to support a third dimension...
B = np.zeros((3,8))

for c in conn:
    xIe = nodes[c,:]
    Ke = np.zeros((8,8))
    for q in q4:
        dN = gradshape(q)
        J = np.dot(dN, xIe).T
        dN = np.dot(np.linalg.inv(J), dN)

        #This part just creates the B matrix using some fancy python tricks its not too complicated
        B[0,0::2] = dN[0,:]
        B[1,1::2] = dN[1,:]
        B[2,0::2] = dN[1,:]
        B[2,1::2] = dN[0,:]

        Ke += np.dot(np.dot(B.T,C), B) * np.linalg.det(J)
    
    for i, I in enumerate(c):
        for j, J in enumerate(c):
            K[2*I,2*J] += Ke[2*i, 2*j]
            K[2*I+1, 2*J] += Ke[2*i+1, 2*j]
            K[2*I+1, 2*J+1] += Ke[2*i+1, 2*j+1]
            K[2*I, 2*J+1] += Ke[2*i,2*j+1]

print("NODAL FORCES AND BOUNDARY CONDITIONS")
f = np.zeros(2*num_nodes)

for i in range(num_nodes):
    if nodes[i, 1] == 0.0:
        K[2*i,:] = 0.0
        K[2*i+1,:] = 0.0
        K[2*i, 2*i] = 1.0
        K[2*i+1, 2*i+1] = 1.0

    #Define the load that is placed at the boundary
    if nodes[i,1] == mesh_ly:
        x = nodes[i, 0]
        f[2*i+1] = 80.0
        if x == 0.0 or x == mesh_lx:
            f[2*i+1] *= 0.5

print("SOLVER STEP")
u = np.linalg.solve(K,f)
print('max u=', max(u))

print("PLOTTING STEP")
ux = np.reshape(u[0::2], (mesh_ny,mesh_nx))
uy = np.reshape(u[1::2], (mesh_ny, mesh_nx))

xvec = []
yvec = []
res = []

for i in range(mesh_nx):
    for j in range(mesh_ny):
        xvec.append(i*mesh_hx + ux[j,i])
        yvec.append(j*mesh_hy + uy[j,i])
        res.append(uy[j,i])

t = plt.tricontourf(xvec, yvec, res, levels=14, cmap=plt.cm.jet)
plt.scatter(xvec, yvec, marker='o', c='b', s=2)
plt.grid()
plt.colorbar(t)
plt.axis('equal')
plt.show()










