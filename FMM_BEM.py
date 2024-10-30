"""
Fast Multipole Method - Boundary Element Method (FMM-BEM)
========================================================

Implementation of the Fast Multipole Method for accelerating
Boundary Element Method calculations for the Laplace equation.

    u_xx + u_yy = 0

with mixed (Dirichlet, Neumann) boundary conditions.
"""

import numpy as np
import meshio
from quadtree import Quadtree
from numpy import log, arctan2, pi
from numpy.linalg import norm
from scipy.special import comb  
import matplotlib.pyplot as plt

#%% Read file and create quadtree

def read_geo_and_create_quadtree(fname, dir_groups, neu_groups, max_points_quad, max_depth):
    """Read the geometry from a Gmsh file with physical groups and create Quadtree structure
    
    Parameters
    ----------
    fname : str
        Path to the mesh file.
    dir_groups : list
        List with the number of the physical groups associated
        with Dirichlet boundary conditions.
    neu_groups : list
        List with the number of the physical groups associated
        with Neumann boundary conditions.
    max_points_quad : int
        Number of max. points per division in the Quadtree.
    max_depth : int
        Number of max. depth allowed in the Quadtree structure.
    
    Returns
    -------
    mesh : meshio Mesh object
        Mesh object.
    coords : ndarray, float
        Coordinates for the endpoints of the elements in the boundary.
    elems : ndarray, int
        Connectivity for the elements.
    x_m : ndarray, float
        Horizontal component of the midpoint of the elements.
    y_m : ndarray, float
        Vertical component of the midpoint of the elements.
    id_dir : list
        Identifiers for elements with Dirichlet boundary conditions.
    id_neu : list
        Identifiers for elements with Neumann boundary conditions.
    quadtree : Object
        Quadtree Object for spatial partitioning.
    boundary_points : ndarray
        Array of boundary points computed for the Quadtree.
    """
    # read file
    mesh = meshio.read(fname)
    
    # obtain elements with Dirichlet and Neumann conditions
    elems_dir = np.vstack([mesh.cells[k].data for k in dir_groups])
    if neu_groups is None:
        elems_neu = np.array([])
        elems = elems_dir.copy()
    else:
        elems_neu = np.vstack([mesh.cells[k].data for k in neu_groups])
        elems = np.vstack((elems_dir, elems_neu))
    
    # Obtain boundary nodes
    bound_nodes = list(set(elems.flatten()))
    coords = mesh.points[bound_nodes, :2]  # 2D
    
    # Compute middle points of the elements 
    x_m, y_m = 0.5 * (coords[elems[:, 0]] + coords[elems[:, 1]]).T
    
    # Idx Cfs
    id_dir = range(elems_dir.shape[0])
    id_neu = range(elems_dir.shape[0], elems_dir.shape[0] + elems_neu.shape[0])
    
    # Extract boundary points
    boundary_cells = mesh.cells_dict.get("line", [])
    boundary_point_indices = set(boundary_cells.flatten())
    boundary_points = mesh.points[list(boundary_point_indices), :2]  #  2D
    
    # lims of quadtree
    xmin, ymin = boundary_points.min(axis=0)
    xmax, ymax = boundary_points.max(axis=0)
    dx = (xmax - xmin) * 0.05
    dy = (ymax - ymin) * 0.05
    xmin, ymin = xmin - dx, ymin - dy
    xmax, ymax = xmax + dx, ymax + dy
    
    # Creating Quadtree
    bounds = [xmin, ymin, xmax, ymax]
    quadtree = Quadtree(bounds, boundary_points, max_points=max_points_quad, max_depth=max_depth)

    return mesh, coords, elems, x_m, y_m, id_dir, id_neu, quadtree, boundary_points

#%% Upward pass

def upward_pass(node, order):
    """
    Performs upward propagation in the quadtree, accumulating multipole moments.
    """
    if not node.children: # If the node has not children, Its a leaf         
        center = node.center
        node.multipole_moments = compute_multipole_moments(node.points, center, order) 
    else: #if not, accumulates the moments of its children using M2M
        moments = np.zeros(order + 1, dtype=complex)
        for child in node.children:
            child_moments = upward_pass(child, order)
            moments += m2m_translation_c(child_moments, child.center, node.center, order)
        node.multipole_moments = moments
    return node.multipole_moments

def compute_multipole_moments(points, center, order):
    """
    Calculate the multipolar moments for the given points in relation with the node center
    """
    moments = np.zeros(order + 1, dtype=complex)
    for x, y in points:
        dx = x - center[0]
        dy = y - center[1]
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            r = 1e-10  #singularity?
        theta = np.arctan2(dy, dx)
        # logaritmic kernel
        for p in range(order + 1):
            moments[p] += r**p * np.exp(-1j * p * theta)
    return moments




#%% Downward pass

def downward_pass(node, order):
    """
    Propagates the local expansions downwards the quadtree using M2L and L2L. 
    Calculates the list of interactions in each node
    """
    results = {}  # dict of results for checking
    interaction_list = node.compute_interaction_list()


    if node.children: 
        node_center = node.center
        # Realizing L2L translation for each child of the actual node
        for child in node.children:
            child_center = child.center
            dx = child_center[0] - node_center[0]
            dy = child_center[1] - node_center[1]

            # L2L from the parent to the child
            child.local_expansion = l2l_translation_c(node.local_expansion, dx, dy, order)

            # recursive to keep propagation
            results[child] = downward_pass(child, order)

    # Computes the contribution of the well-separated cells with M2L
    for source_node in interaction_list:
        dx = node.center[0] - source_node.center[0]
        dy = node.center[1] - source_node.center[1]
        node.local_expansion += m2l_translation_c(source_node.multipole_moments, dx, dy, order)

    results[node] = node.local_expansion
    return results




def l2l_translation_c(local_expansion_parent, dx, dy, order):
    """
    Translation Local-to-Local (L2L) en coordenadas cartesianas.
    """
    local_expansion_child = np.zeros(order + 1, dtype=complex)
    
    for n in range(order + 1):
        for k in range(n + 1):
            if k < len(local_expansion_parent):
                binomial = comb(n, k, exact=True)
                local_expansion_child[n] += (
                    binomial * local_expansion_parent[k] * 
                    (dx + 1j * dy)**(n - k)
                )
    
    return local_expansion_child

def m2l_translation_c(multipole_moments, dx, dy, order):
    """
    Translation Multipole-to-Local (M2L) en coordenadas cartesianas.
    """
    local_expansion = np.zeros(order + 1, dtype=complex)
    
    for n in range(order + 1):
        for k in range(order + 1):
            if k < len(multipole_moments):
                local_expansion[n] += (
                    multipole_moments[k] * 
                    (-1)**k * (dx - 1j * dy)**(-(k + n + 1))
                )
    
    return local_expansion

def m2m_translation_c(child_moments, child_center, parent_center, order):
    """
    Translation moment-to-moment (M2M) en coordenadas cartesianas,
    desde un nodo hijo hacia el nodo padre.
    """
    dx = child_center[0] - parent_center[0]
    dy = child_center[1] - parent_center[1]

    # Expansión en serie de Taylor
    translated_moments = np.zeros(order + 1, dtype=complex)
    for p in range(order + 1):
        for k in range(p + 1):
            if k < len(child_moments):
                translated_moments[p] += child_moments[k] * (dx + 1j * dy)**(p - k)
    
    return translated_moments













#%% Processing 

def influence_coeff(elem, coords, pt_col):
    """Compute influence coefficients

    Parameters
    ----------
    elems : ndarray, int
        Connectivity for the elements.
    coords : ndarray, float
        Coordinates for the nodes.
    pt_col : ndarray
        Coordinates of the colocation point.

    Returns
    -------
    G_coeff : float
        Influence coefficient for flows.
    H_coeff : float
        Influence coefficient for primary variable.
    """
    dcos = coords[elem[1]] - coords[elem[0]]
    dcos = dcos / norm(dcos)
    rotmat = np.array([[dcos[1], -dcos[0]],
                       [dcos[0], dcos[1]]])
    r_A = rotmat.dot(coords[elem[0]] - pt_col)
    r_B = rotmat.dot(coords[elem[1]] - pt_col)
    theta_A = arctan2(r_A[1], r_A[0])
    theta_B = arctan2(r_B[1], r_B[0])
    if norm(r_A) <= 1e-6:
        G_coeff = r_B[1]*(log(norm(r_B)) - 1) + theta_B*r_B[0]
    elif norm(r_B) <= 1e-6:
        G_coeff = -(r_A[1]*(log(norm(r_A)) - 1) + theta_A*r_A[0])
    else:
        G_coeff = r_B[1]*(log(norm(r_B)) - 1) + theta_B*r_B[0] -\
                  (r_A[1]*(log(norm(r_A)) - 1) + theta_A*r_A[0])
    H_coeff = theta_B - theta_A
    return -G_coeff/(2*pi), H_coeff/(2*pi)

def assem_fmm(coords, elems, quadtree, order):
    """
    Assembly matrices for the BEM problem with FMM
    
    Parameters
    ----------
    coords : ndarray, float
        Coordinates for the nodes.
    elems : ndarray, int
        Connectivity for the elements.
    quadtree : Quadtree object
        Quadtree structure containing the elements and points.
    order : int
        Order of the multipole expansions.

    Returns
    -------
    Gmat : ndarray, float
        Influence matrix for the flow.
    Hmat : ndarray, float
        Influence matrix for primary variable.
    """
    nelems = elems.shape[0]
    Gmat = np.zeros((nelems, nelems))
    Hmat = np.zeros((nelems, nelems))

    upward_pass(quadtree, order)
    downward_results = downward_pass(quadtree, order)

    # Assembling the influence matrices 
    for ev_cont, elem1 in enumerate(elems):
        for col_cont, elem2 in enumerate(elems):
            pt_col = np.mean(coords[elem2], axis=0)

            # obtaining the nodes of each element
            node_info1 = quadtree.find_point_location(np.mean(coords[elem1], axis=0))
            node_info2 = quadtree.find_point_location(pt_col)
            
            node1 = node_info1['leaf']
            node2 = node_info2['leaf']

            # Computing directly the interaction if the nodes are adjacents 
            if node1 == node2 or node1.is_adjacent(node2):
                if ev_cont == col_cont:
                    # self-interaction
                    L = np.linalg.norm(coords[elem1[1]] - coords[elem1[0]])
                    Gmat[ev_cont, ev_cont] += -L / (2 * np.pi) * (np.log(L / 2) - 1)
                    Hmat[ev_cont, ev_cont] += -0.5
                else:
                    # Direct interaction between near element 
                    Gij, Hij = influence_coeff(elem1, coords, pt_col)
                    Gmat[ev_cont, col_cont] += Gij
                    Hmat[ev_cont, col_cont] += Hij
            else:
                 # Far interaction using M2L 
                 interaction_list = node1.compute_interaction_list()
                 if node2 in interaction_list:
                     dx = node2.center[0] - node1.center[0]
                     dy = node2.center[1] - node1.center[1]
                     Gmat[ev_cont, col_cont] += np.real(m2l_translation_c(node2.multipole_moments, dx, dy, order)[0]) #???
                     Hmat[ev_cont, col_cont] += np.real(l2l_translation_c(node2.local_expansion, dx, dy, order)[0]) #????
    return Gmat, Hmat


#%% post process

def eval_sol(ev_coords, coords, elems, u_boundary, q_boundary):
    """Evaluate the solution in a set of points

    Parameters
    ----------
    ev_coords : ndarray, float
        Coordinates of the evaluation points.
    coords : ndarray, float
        Coordinates for the nodes.
    elems : ndarray, int
        Connectivity for the elements.
    u_boundary : ndarray, float
        Primary variable in the nodes.
    q_boundary : ndarray, float
        Flows in the nodes.

    Returns
    -------
    solution : ndarray, float
        Solution evaluated in the given points.
    """
    npts = ev_coords.shape[0]
    solution = np.zeros(npts)
    for k in range(npts):
        for ev_cont, elem in enumerate(elems):        
            pt_col = ev_coords[k]
            G, H = influence_coeff(elem, coords, pt_col)
            solution[k] += u_boundary[ev_cont]*H - q_boundary[ev_cont]*G
    return solution

def surf_plot(mesh, tri_group, field, ax=None):
    """Plot a field as a deformed surface

    Parameters
    ----------
    mesh : Meshio mesh object
        Mesh object.
    tri_group : int
        Identifier for the physical group of the triangles
    field : ndarray, float
        Field to visualize.
    ax : Matplotlib axes, optional
        Axes for the plot, by default None.

    Returns
    -------
    ax : Matplotlib axes
        Axes where the plot was created.
    """
    pt_x, pt_y = mesh.points[:, :2].T
    tris = mesh.cells[tri_group].data
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = plt.gcf()
    ax.plot_trisurf(pt_x, pt_y, field, triangles=tris, cmap="RdBu", lw=0.1,
                    edgecolor="#3c3c3c")
    return ax
