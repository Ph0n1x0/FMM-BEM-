import numpy as np

class Quadtree:
    def __init__(self, bounds, points, max_points=1, depth=0, max_depth=10, parent=None):
        self.bounds = bounds  # [xmin, ymin, xmax, ymax]
        self.points = [p for p in points if self.point_in_bounds(p, bounds)]
        self.max_points = max_points
        self.depth = depth
        self.max_depth = max_depth
        self.parent = parent
        self.children = []
        self.multipole_moments = np.zeros(max_points + 1, dtype=complex)
        self.local_expansion = np.zeros(max_points + 1, dtype=complex)
        
        # Calcula el centro del nodo
        xmin, ymin, xmax, ymax = self.bounds
        self.center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        if len(self.points) > max_points and depth < max_depth:
            self.subdivide()

    def subdivide(self):
        xmin, ymin, xmax, ymax = self.bounds
        width = xmax - xmin
        height = ymax - ymin
        size = max(width, height)
        xmid, ymid = self.center

        quadrants = [
            [xmin, ymin, xmid, ymid],
            [xmid, ymin, xmin + size, ymid],
            [xmin, ymid, xmid, ymin + size],
            [xmid, ymid, xmin + size, ymin + size]
        ]

        for quad in quadrants:
            qpoints = [p for p in self.points if self.point_in_bounds(p, quad)]
            if qpoints:
                child = Quadtree(quad, qpoints, self.max_points, self.depth + 1, self.max_depth, self)
                self.children.append(child)

    def point_in_bounds(self, point, bounds):
        x, y = point
        xmin, ymin, xmax, ymax = bounds
        return xmin <= x < xmax and ymin <= y < ymax

    def find_point_location(self, point):
        """
        Encuentra la hoja y el camino de nodos padres para un punto dado.
        
        Parameters:
        -----------
        point : tuple
            Coordenadas (x, y) del punto a buscar.
        
        Returns:
        --------
        dict
            Un diccionario con las claves 'leaf' (nodo hoja), 'path' (camino de nodos),
            y 'depth' (profundidad del nodo hoja).
        """
        def _find_recursive(node, point, path):
            path.append(node)
            
            if not node.children:  # Si es una hoja
                return {
                    'leaf': node,
                    'path': path,
                    'depth': len(path) - 1
                }
            
            for child in node.children:
                if child.point_in_bounds(point, child.bounds):
                    return _find_recursive(child, point, path)
        
        return _find_recursive(self, point, path=[])