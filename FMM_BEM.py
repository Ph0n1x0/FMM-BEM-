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
import matplotlib.pyplot as plt
import math

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

def compute_multipole_moments(points, cell_center, max_order):
    """
    Calcula los momentos multipolares hasta un orden máximo dado para una celda en 2D.

    Parámetros:
    - points: Lista o array de coordenadas de los puntos en la celda, de tamaño (N, 2).
    - charges: Array de valores de carga (o intensidad del campo) en cada punto, de tamaño (N,).
    - cell_center: Coordenadas del centro de la celda, en forma (2,).
    - max_order: Orden máximo para los momentos multipolares.

    Retorna:
    - moments: Array de momentos multipolares hasta el orden max_order, de tamaño (max_order+1,).
    """
    # Convertir points en un array de numpy y asegurar que es 2D
    points = np.atleast_2d(np.array(points))

    # Verificar que hay puntos y que tiene la forma correcta
    if points.shape[0] == 0 or points.shape[1] != 2:
        return np.zeros(max_order + 1, dtype=complex)  # Retornar momentos nulos si no hay puntos o la forma es incorrecta


    # Convertir el centro de la celda y los puntos a coordenadas complejas
    z_c = cell_center[0] + 1j * cell_center[1]
    z_points = points[:, 0] + 1j * points[:, 1]
    
    # Inicializar los momentos
    moments = np.zeros(max_order + 1, dtype=complex)

    # Calcular cada momento hasta el orden max_order
    for k in range(max_order + 1):
        # Ik(z) = z^k / k!
        I_k = ((z_points - z_c) ** k) / math.factorial(k)
        # Momento M_k = suma de (I_k * carga)
        moments[k] = np.sum(I_k )

    return moments

def upward_pass(node, order):
    """
    Realiza la propagación hacia arriba en el quadtree, acumulando momentos multipolares.
    """
    if not node.children:  # Si es una hoja
        center = node.center
        node.multipole_moments = compute_multipole_moments(node.points, center, order)
    else:
        moments = np.zeros(order + 1, dtype=complex)
        for child in node.children:
            child_moments = upward_pass(child, order)
            # Aplicar la traducción M2M para transformar los momentos del hijo al centro del nodo actual
            moments += m2m_translation(child_moments, child.center, node.center, order)
        node.multipole_moments = moments
    return node.multipole_moments


#%% Downward pass

def m2l_translation(multipole_moments, dx, dy, order):
    local_expansion = np.zeros(order, dtype=complex)
    distance = complex(dx, dy)

    for l in range(order):
        local_expansion[l] = sum(
            (-1) ** k * multipole_moments[k] * (distance ** (l + k)) / math.factorial(l + k)
            for k in range(order)
        )
    return local_expansion

def l2l_translation(local_expansion_parent, dx, dy, order):
    local_expansion_child = np.zeros(order, dtype=complex)
    distance = complex(dx, dy)

    for l in range(order):
        local_expansion_child[l] = sum(
            local_expansion_parent[k] * (distance ** (l - k)) / math.factorial(l - k)
            for k in range(l + 1)
        )
    return local_expansion_child

def downward_pass(node, order):
    # Inicializar la expansión local
    node.local_expansion = np.zeros(order, dtype=complex)

    # Realizar la traducción M2L para los nodos de la lista de interacción
    interaction_list = node.compute_interaction_list()
    for interacting_node in interaction_list:
        translation = m2l_translation(
            interacting_node.multipole_moments, 
            node.center[0] - interacting_node.center[0], 
            node.center[1] - interacting_node.center[1], 
            order
        )
        node.local_expansion += translation

    # Si el nodo tiene hijos, aplicar L2L para los hijos
    for child in node.children:
        child.local_expansion = l2l_translation(
            node.local_expansion,
            child.center[0] - node.center[0],
            child.center[1] - node.center[1],
            order
        )

    # Aplicar recursivamente el descenso en los hijos
    for child in node.children:
        downward_pass(child, order)

def m2m_translation(child_moments, child_center, parent_center, max_order):
    """
    Traduce los momentos de un nodo hijo al centro de un nodo padre usando la expansión M2M.

    Parámetros:
    - child_moments: Array de momentos multipolares del nodo hijo, de tamaño (max_order + 1,).
    - child_center: Coordenadas del centro del nodo hijo en forma (2,).
    - parent_center: Coordenadas del centro del nodo padre en forma (2,).
    - max_order: Orden máximo de los momentos.

    Retorna:
    - translated_moments: Array de momentos multipolares traducidos al centro del nodo padre.
    """
    # Convertir centros a coordenadas complejas
    z_child = child_center[0] + 1j * child_center[1]
    z_parent = parent_center[0] + 1j * parent_center[1]
    delta_z = z_child - z_parent  # Desplazamiento entre los centros

    # Inicializar los momentos traducidos
    translated_moments = np.zeros(max_order + 1, dtype=complex)

    # Realizar la traducción M2M
    for k in range(max_order + 1):
        for j in range(k + 1):
            factor = (delta_z ** (k - j)) / math.factorial(k - j)
            translated_moments[k] += child_moments[j] * factor

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

def assem_FMM(coords, elems, quadtree, order):
    nelems = elems.shape[0]
    Gmat = np.zeros((nelems, nelems))
    Hmat = np.zeros((nelems, nelems))
    
    # Realizar las fases ascendente y descendente del FMM
    upward_pass(quadtree, order)
    downward_pass(quadtree, order)
    
    for i, elem1 in enumerate(elems):
        pt1 = np.mean(coords[elem1], axis=0)
        for j, elem2 in enumerate(elems):
            pt2 = np.mean(coords[elem2], axis=0)
            
            # Localizar las celdas en el quadtree
            node_info1 = quadtree.find_point_location(pt1)
            node_info2 = quadtree.find_point_location(pt2)
            node1 = node_info1['leaf']
            node2 = node_info2['leaf']
            
            if node1 == node2:
                # Interacciones cercanas
                if i == j:
                    # Auto-interacción
                    L = norm(coords[elem1[1]] - coords[elem1[0]])
                    Gmat[i, j] = (- (L/(2*pi))*(log(L/2) - 1))
                    Hmat[i, j] = (- 0.5)
                    
                else:
                    # Interacción directa
                    d = np.linalg.norm(pt1 - pt2)
                    Gmat[i, j] = (1 / (2 * np.pi * d))
                    Hmat[i, j] = (-1 / (2 * np.pi * d ** 2))
            elif  node1.is_adjacent(node2):
                if i == j:
                    # Auto-interacción
                    L = norm(coords[elem1[1]] - coords[elem1[0]])
                    Gmat[i, j] = (- (L/(2*pi))*(log(L/2) - 1))
                    Hmat[i, j] = (- 0.5)
                    
                else:
                    # Interacción directa
                    d = np.linalg.norm(pt1 - pt2)
                    Gmat[i, j] = (1 / (2 * np.pi * d))
                    Hmat[i, j] = (-1 / (2 * np.pi * d ** 2))
            else:
                # Interacciones lejanas usando FMM 
                dx = node1.center[0] - node2.center[0]
                dy = node1.center[1] - node2.center[1]
                local_expansion = m2l_translation(node2.multipole_moments, dx, dy, order)
                z = complex(pt1[0] - node1.center[0], pt1[1] - node1.center[1])
                potential = sum(coeff * (z**k) for k, coeff in enumerate(local_expansion))
                Gmat[i, j] += np.real(potential) / (2 * np.pi)
                Hmat[i, j] += np.imag(potential) / (2 * np.pi)

    
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
