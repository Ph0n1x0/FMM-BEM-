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
from scipy.sparse.linalg import gmres, LinearOperator
from numpy import log, sin, cos, arctan2, pi, mean
from numpy.linalg import norm, solve
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
    # Leer archivo de malla usando meshio
    mesh = meshio.read(fname)
    
    # Obtener elementos con condiciones Dirichlet y Neumann
    elems_dir = np.vstack([mesh.cells[k].data for k in dir_groups])
    if neu_groups is None:
        elems_neu = np.array([])
        elems = elems_dir.copy()
    else:
        elems_neu = np.vstack([mesh.cells[k].data for k in neu_groups])
        elems = np.vstack((elems_dir, elems_neu))
    
    # Obtener nodos de frontera
    bound_nodes = list(set(elems.flatten()))
    coords = mesh.points[bound_nodes, :2]  # Solo usamos las coordenadas 2D
    
    # Calcular puntos medios de los elementos
    x_m, y_m = 0.5 * (coords[elems[:, 0]] + coords[elems[:, 1]]).T
    
    # Identificar elementos con condiciones Dirichlet y Neumann
    id_dir = range(elems_dir.shape[0])
    id_neu = range(elems_dir.shape[0], elems_dir.shape[0] + elems_neu.shape[0])
    
    # Extraer puntos de contorno y calcular límites para el Quadtree
    boundary_cells = mesh.cells_dict.get("line", [])
    boundary_point_indices = set(boundary_cells.flatten())
    boundary_points = mesh.points[list(boundary_point_indices), :2]  # Solo en 2D
    
    # Calcular límites de los puntos de contorno con una expansión del 5%
    xmin, ymin = boundary_points.min(axis=0)
    xmax, ymax = boundary_points.max(axis=0)
    dx = (xmax - xmin) * 0.05
    dy = (ymax - ymin) * 0.05
    xmin, ymin = xmin - dx, ymin - dy
    xmax, ymax = xmax + dx, ymax + dy
    
    # Crear el Quadtree
    bounds = [xmin, ymin, xmax, ymax]
    quadtree = Quadtree(bounds, boundary_points, max_points=max_points_quad, max_depth=max_depth)
    
    # Retornar toda la información
    return mesh, coords, elems, x_m, y_m, id_dir, id_neu, quadtree, boundary_points

#%% Upward pass

def upward_pass(node, order):
    """
    Realiza la propagación hacia arriba en el quadtree, acumulando momentos multipolares.
    """
    if not node.children:
        # Si es un nodo hoja, calcula los momentos multipolares.
        center = node.center
        node.multipole_moments = compute_multipole_moments(node.points, center, order)
    else:
        # Si no es un nodo hoja, acumula los momentos de los hijos usando M2M (momento a momento).
        moments = np.zeros(order + 1, dtype=complex)
        for child in node.children:
            child_moments = upward_pass(child, order)
            # Traducción de momentos desde el hijo al nodo padre (M2M)
            moments += m2m_translation(child_moments, child.center, node.center, order)
        node.multipole_moments = moments
    return node.multipole_moments

def compute_multipole_moments(points, center, order):
    """
    Calcula los momentos multipolares para los puntos dados en relación al centro del nodo.
    """
    moments = np.zeros(order + 1, dtype=complex)
    for x, y in points:
        dx = x - center[0]
        dy = y - center[1]
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            r = 1e-10  # Evitar singularidades
        theta = np.arctan2(dy, dx)
        # Cálculo de los momentos usando el kernel logarítmico
        for p in range(order + 1):
            moments[p] += r**p * np.exp(-1j * p * theta)
    return moments

def m2m_translation(child_moments, child_center, parent_center, order):
    """
    Traducción de momento a momento (M2M) de un nodo hijo al nodo padre.
    """
    dx = child_center[0] - parent_center[0]
    dy = child_center[1] - parent_center[1]
    r = np.sqrt(dx**2 + dy**2)
    if r == 0:
        r = 1e-10  # Evitar divisiones por cero
    theta = np.arctan2(dy, dx)

    # Expansión de Taylor para la traducción de momentos multipolares de hijo a padre.
    translated_moments = np.zeros(order + 1, dtype=complex)
    for p in range(order + 1):
        for k in range(p + 1):
            translated_moments[p] += child_moments[k] * (r**(p-k)) * np.exp(1j * (p-k) * theta)
    
    return translated_moments


#%% Downward pass

def downward_pass(node, order):
    """
    Propaga las expansiones locales hacia abajo en el quadtree usando M2L y L2L.
    Calcula la lista de interacción en cada nodo.
    """
    results = {}  # Diccionario para almacenar las expansiones locales por nodo
    
    # Calcular la lista de interacción para el nodo actual
    interaction_list = node.compute_interaction_list()

    # Si el nodo tiene hijos, continuamos con la propagación descendente
    if node.children:
        node_center = node.center

        # Para cada hijo del nodo actual, realizamos la traducción L2L
        for child in node.children:
            child_center = child.center
            dx = child_center[0] - node_center[0]
            dy = child_center[1] - node_center[1]

            # Realizamos la traducción L2L (Local a Local) desde el nodo padre al hijo
            child.local_expansion = l2l_translation(node.local_expansion, dx, dy, order)

            # Llamada recursiva para continuar la propagación hacia los hijos
            results[child] = downward_pass(child, order)

    # Calcular la contribución de las células bien separadas mediante M2L
    for source_node in interaction_list:
        dx = node.center[0] - source_node.center[0]
        dy = node.center[1] - source_node.center[1]
        # Realizamos la traducción M2L (Momento a Local) usando la expansión multipolar del nodo fuente
        node.local_expansion += m2l_translation(source_node.multipole_moments, dx, dy, order)

    # Almacena la expansión local del nodo actual y retorna los resultados
    results[node] = node.local_expansion
    return results

def l2l_translation(local_expansion_parent, dx, dy, order):
    """
    Traducción Local-to-Local (L2L).
    
    Parameters:
    -----------
    local_expansion_parent : ndarray
        Expansión local del nodo padre
    dx, dy : float
        Desplazamiento relativo
    order : int
        Orden de la expansión
    
    Returns:
    --------
    ndarray
        Expansión local traducida
    """
    r = np.sqrt(dx**2 + dy**2)
    if r < 1e-10:
        return local_expansion_parent.copy()
        
    theta = np.arctan2(dy, dx)
    local_expansion_child = np.zeros(order + 1, dtype=complex)
    
    for n in range(order + 1):
        for k in range(n + 1):
            if k < len(local_expansion_parent):
                binomial = comb(n, k, exact=True)  
                local_expansion_child[n] += (
                    binomial * local_expansion_parent[k] * 
                    (r**(n-k)) * np.exp(1j * (n-k) * theta)
                )
    
    return local_expansion_child

def m2l_translation(multipole_moments, dx, dy, order):
    """
    Traducción Multipole-to-Local (M2L).
    """
    r = np.sqrt(dx**2 + dy**2)
    if r < 1e-10:
        return np.zeros(order + 1, dtype=complex)
        
    theta = np.arctan2(dy, dx)
    local_expansion = np.zeros(order + 1, dtype=complex)
    
    for n in range(order + 1):
        for k in range(order + 1):
            if k < len(multipole_moments):
                local_expansion[n] += (
                    multipole_moments[k] * 
                    (-1)**k * (1/r)**(k+n+1) * 
                    np.exp(-1j * (k+n) * theta)
                )
    
    return local_expansion



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
    """Assembly matrices for the BEM problem with FMM
    
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

    # Paso ascendente: computamos las expansiones multipolares hacia arriba en el quadtree
    upward_pass(quadtree, order)

    # Paso descendente: propagamos las expansiones locales hacia abajo
    downward_results = downward_pass(quadtree, order)

    # Ensamblamos las matrices de influencia
    for ev_cont, elem1 in enumerate(elems):
        for col_cont, elem2 in enumerate(elems):
            pt_col = np.mean(coords[elem2], axis=0)

            # Usamos find_point_location para obtener los nodos correspondientes a cada elemento
            node_info1 = quadtree.find_point_location(np.mean(coords[elem1], axis=0))
            node_info2 = quadtree.find_point_location(pt_col)
            
            node1 = node_info1['leaf']
            node2 = node_info2['leaf']

            # Si los nodos son adyacentes, calculamos la interacción directamente
            if node1 == node2 or node1.is_adjacent(node2):
                if ev_cont == col_cont:
                    # Interacción con sí mismo (diagonal)
                    L = np.linalg.norm(coords[elem1[1]] - coords[elem1[0]])
                    Gmat[ev_cont, ev_cont] = -L / (2 * np.pi) * (np.log(L / 2) - 1)
                    Hmat[ev_cont, ev_cont] = -0.5
                else:
                    # Interacción directa entre elementos cercanos
                    Gij, Hij = influence_coeff(elem1, coords, pt_col)
                    Gmat[ev_cont, col_cont] = Gij
                    Hmat[ev_cont, col_cont] = Hij
            else:
                # Interacción lejana: utilizamos M2L (multipole to local)
                interaction_list = node1.compute_interaction_list()
                if node2 in interaction_list:
                    dx = node2.center[0] - node1.center[0]
                    dy = node2.center[1] - node1.center[1]
                    # Traducción multipolar M2L para interacciones lejanas
                    Gmat[ev_cont, col_cont] += np.sum(m2l_translation(node2.multipole_moments, dx, dy, order))
                    Hmat[ev_cont, col_cont] += np.sum(l2l_translation(node2.local_expansion, dx, dy, order))
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
