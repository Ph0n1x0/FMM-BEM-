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

#create quadtree

def create_quad_tree_w_msh_file(file_name, max_points_quad, max_depth):
    """Read the .msh file from gmsh and create Quadtree structure
    Parameters
    ----------
    file_name : str
        Path to the mesh file.
    max_points_quad : int
        Number of max. points per division
    max_depth : int
        Number of max. depth allowed in the quadtree structure
    Returns
    -------
    Quadtree : Object
        Quadtree Object
    boundary_points : ndarray
        Array of boundary points computed 
    """
    
    # Read mesh file
    mesh = meshio.read(file_name)
    
    # Extract points and boundary cells
    points = mesh.points
    boundary_cells = mesh.cells_dict.get("line", [])
    boundary_point_indices = set(boundary_cells.flatten())
    boundary_points = points[list(boundary_point_indices), :2]  # 2D
    
    # Calculate limits with expansion
    xmin, ymin = boundary_points.min(axis=0)
    xmax, ymax = boundary_points.max(axis=0)
    
    # Expand bounding box by 5%
    dx = (xmax - xmin) * 0.05
    dy = (ymax - ymin) * 0.05
    xmin, ymin = xmin - dx, ymin - dy
    xmax, ymax = xmax + dx, ymax + dy
    
    # Create quadtree
    bounds = [xmin, ymin, xmax, ymax]
    quadtree = Quadtree(bounds, boundary_points, max_points=max_points_quad, max_depth=max_depth)
    
    return quadtree, boundary_points

#upward and downward

def upward_pass(node, order):
    """
    Realiza la propagación hacia arriba en el quadtree.
    """
    if not node.children:
        # Si es un nodo hoja, calcula los momentos multipolares.
        center = node.center
        node.multipole_moments = compute_multipole_moments(node.points, center, order)
    else:
        # Si no es hoja, acumula los momentos de los hijos.
        moments = np.zeros(order + 1, dtype=complex)
        for child in node.children:
            child_moments = upward_pass(child, order)
            dx = (child.bounds[0] + child.bounds[2]) / 2 - (node.bounds[0] + node.bounds[2]) / 2
            dy = (child.bounds[1] + child.bounds[3]) / 2 - (node.bounds[1] + node.bounds[3]) / 2
            r = np.sqrt(dx**2 + dy**2)
            if r == 0:
                r=1e-10
            theta = np.arctan2(dy, dx)
            for p in range(order + 1):
                moments[p] += child_moments[p] * np.exp(1j * p * theta) * r**p
        node.multipole_moments = moments
    return node.multipole_moments

def downward_pass(node, order):
    """
    Propaga las expansiones locales hacia abajo en el quadtree y retorna las expansiones calculadas.
    """
    results = {}  # Diccionario para almacenar las expansiones locales por nodo

    if node.children:  # Si el nodo tiene hijos

        node_center = node.center

        # Propagamos las expansiones hacia cada hijo.
        for child in node.children:
            child_center = child.center

            dx = child_center[0] - node_center[0]
            dy = child_center[1] - node_center[1]

            # Verificamos que r no sea cero para evitar división por cero
            r = np.sqrt(dx**2 + dy**2)
            if r == 0:
                r = 1e-10  # Ajuste para evitar problemas numéricos

            # Realizamos la traducción M2L
            child.local_expansion = m2l_translation(node.multipole_moments, dx, dy, order)

            # Llamada recursiva para continuar la propagación
            results[child] = downward_pass(child, order)

    # Almacena la expansión local del nodo actual y retorna los resultados
    results[node] = node.local_expansion
    return results 

#moments and interaction functions

def compute_multipole_moments(points, center, order):
    """
    Calcula los momentos multipolares para los puntos dados.
    """
    moments = np.zeros(order + 1, dtype=complex)
    for x, y in points:
        dx = x - center[0]
        dy = y - center[1]
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            r=1e-10
        theta = np.arctan2(dy, dx)
        for p in range(order + 1):
            moments[p] += r**p * np.exp(-1j * p * theta)
    return moments

def l2l_translation(local_expansion_parent, dx, dy, order):
    """
    Traducción Local-to-Local (L2L) según la ecuación (27).
    Propaga la expansión local del nodo padre al nodo hijo.
    """
    r = np.sqrt(dx**2 + dy**2)
    if r == 0:
        r=1e-10
    theta = np.arctan2(dy, dx)
    local_expansion_child = np.zeros(order + 1, dtype=complex)

    # Aplicamos la ecuación (27) para la traducción L2L.
    for i in range(order + 1):
        for k in range(order + 1 - i):
            local_expansion_child[i] += (
                local_expansion_parent[i + k] * (r**k) * np.exp(1j * k * theta)
            )
    return local_expansion_child

def m2l_translation(multipole_moments, dx, dy, order):
    """
    Traducción Moment-to-Local (M2L) para pasar los momentos multipolares a expansiones locales.
    """
    r = np.sqrt(dx**2 + dy**2)
    if r == 0:
        r=1e-10
    theta = np.arctan2(dy, dx)
    local_expansion = np.zeros(order + 1, dtype=complex)

    # Aplicamos la traducción M2L.
    for i in range(order + 1):
        for k in range(order + 1):
            local_expansion[i] += (
                multipole_moments[k] * (r**(i - k)) * np.exp(1j * (i - k) * theta)
            )
    return local_expansion

#evaluate field at bounds

def evaluate_field_at_targets(node):
    """
    Evalúa el campo en los puntos de la frontera usando las expansiones locales.
    """
    results = {}
    if node.local_expansion is None:
        node.local_expansion = np.zeros_like(node.multipole_moments)  # Inicialización segura

    for point in node.points:
        field = 0
        for p, moment in enumerate(node.local_expansion):
            dx = point[0] - (node.bounds[0] + node.bounds[2]) / 2
            dy = point[1] - (node.bounds[1] + node.bounds[3]) / 2
            r = np.sqrt(dx**2 + dy**2)

            if r > 0:
                theta = np.arctan2(dy, dx)
                field += moment * (r**p) * np.exp(1j * p * theta)

        results[tuple(point)] = field

    for child in node.children:
        results.update(evaluate_field_at_targets(child))

    return results

#system for GMRES

def create_linear_operator(boundary_points, field_values):
    """
    Crea un LinearOperator basado en los puntos de la frontera y los valores del campo.
    """
    n = len(boundary_points)
    
    def matvec(x):
        """
        Producto matriz-vector usando las expansiones locales.
        """
        result = np.zeros_like(x, dtype=complex)
        for idx, point in enumerate(boundary_points):
            # Evaluación del campo para cada punto
            result[idx] = field_values.get(tuple(point), 0)
        return result
    
    return LinearOperator((n, n), matvec=matvec)

def solve_bem_system(boundary_points, field_values, rhs=None):
    """
    Resuelve el sistema BEM usando GMRES.
    
    :param boundary_points: Lista de puntos en la frontera
    :param field_values: Diccionario con los valores del campo para cada punto
    :param rhs: Vector del lado derecho (condiciones de frontera). Si es None, se usa un vector de unos.
    :return: Solución del sistema
    """
    n = len(boundary_points)
    A = create_linear_operator(boundary_points, field_values)
    
    if rhs is None:
        rhs = np.ones(n, dtype=complex)  # Condiciones de frontera de ejemplo
    
    print("Resolviendo el sistema con GMRES...")
    solution = gmres(A, rhs)
    
    return solution
