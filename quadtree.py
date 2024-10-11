import numpy as np

class Quadtree:
    def __init__(self, bounds, points, max_points=4, depth=0, max_depth=10):
        """
        bounds: [xmin, ymin, xmax, ymax] - Los límites del dominio en el nodo.
        points: Lista de puntos dentro de este nodo.
        max_points: Máximo número de puntos por nodo antes de subdividir.
        depth: Profundidad actual del nodo.
        max_depth: Profundidad máxima permitida para la subdivisión.
        """
        self.bounds = bounds  # [xmin, ymin, xmax, ymax]
        self.points = points  # Lista de puntos en este nodo
        self.max_points = max_points  # Máximo número de puntos antes de subdividir
        self.depth = depth  # Profundidad actual del quadtree
        self.max_depth = max_depth  # Máxima profundidad permitida
        self.children = []  # Hijos del nodo

        # Si hay más puntos de lo permitido y no hemos alcanzado la profundidad máxima, subdividir
        if len(points) > max_points and depth < max_depth:
            self.subdivide()

    def subdivide(self):
        """ Subdivide el nodo en 4 cuadrantes. """
        xmin, ymin, xmax, ymax = self.bounds
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        # Crear los 4 nuevos cuadrantes
        quadrants = [
            [xmin, ymin, xmid, ymid],  # Inferior izquierdo
            [xmid, ymin, xmax, ymid],  # Inferior derecho
            [xmin, ymid, xmid, ymax],  # Superior izquierdo
            [xmid, ymid, xmax, ymax]   # Superior derecho
        ]

        # Agrupar los puntos en sus nuevos cuadrantes
        for quad in quadrants:
            qpoints = [p for p in self.points if self.point_in_bounds(p, quad)]
            if qpoints:
                self.children.append(Quadtree(quad, qpoints, self.max_points, self.depth + 1, self.max_depth))

    def point_in_bounds(self, point, bounds):
        """ Verifica si un punto está dentro de los límites dados. """
        x, y = point
        xmin, ymin, xmax, ymax = bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

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
