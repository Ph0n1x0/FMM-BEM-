import numpy as np

class Quadtree:
    def __init__(self, bounds, points, max_points=1, depth=0, max_depth=10, parent=None):
        self.bounds = bounds  # [xmin, ymin, xmax, ymax]
        self.points = points
        self.max_points = max_points
        self.depth = depth
        self.max_depth = max_depth
        self.parent = parent  # Referencia al nodo padre
        self.children = []  # Lista de nodos hijos
        self.multipole_moments = np.zeros(max_points + 1, dtype=complex)  # Momentos multipolares
        self.local_expansion = np.zeros(max_points + 1, dtype=complex)  # Expansión local inicializada

        if len(points) > max_points and depth < max_depth:
            self.subdivide()

    def subdivide(self):
        """Divide el nodo en 4 cuadrantes y asigna nodos hijos."""
        xmin, ymin, xmax, ymax = self.bounds
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        quadrants = [
            [xmin, ymin, xmid, ymid],
            [xmid, ymin, xmax, ymid],
            [xmin, ymid, xmid, ymax],
            [xmid, ymid, xmax, ymax]
        ]

        for quad in quadrants:
            qpoints = [p for p in self.points if self.point_in_bounds(p, quad)]
            if qpoints:
                child = Quadtree(quad, qpoints, self.max_points, self.depth + 1, self.max_depth, self)
                self.children.append(child)

    def point_in_bounds(self, point, bounds):
        """Verifica si un punto está dentro de los límites."""
        x, y = point
        xmin, ymin, xmax, ymax = bounds
        return xmin <= x < xmax and ymin <= y < ymax

    def query(self, point, radius):
        """ Realiza una búsqueda de puntos cercanos en el Quadtree. """
        found_points = []
        if not self.intersects_circle(point, radius):
            return found_points

        for p in self.points:
            if np.linalg.norm(np.array(p) - np.array(point)) <= radius:
                found_points.append(p)

        for child in self.children:
            found_points.extend(child.query(point, radius))

        return found_points

    def intersects_circle(self, point, radius):
        """ Verifica si un círculo intersecta los límites de este nodo. """
        x, y = point
        xmin, ymin, xmax, ymax = self.bounds
        closest_x = np.clip(x, xmin, xmax)
        closest_y = np.clip(y, ymin, ymax)
        return np.linalg.norm([x - closest_x, y - closest_y]) <= radius


