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
        
        # compute the node center
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
        
        #define quadrants from lower left corner counter-clockwise
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
            elif not qpoints: #define empty quads 
                child = Quadtree(quad,[],self.max_points, self.depth + 1,max_depth=0)
                self.children.append(child)    

    def is_adjacent(self, other):
        """
        Check if another node is adjacent to the actual node
        """
        xmin, ymin, xmax, ymax = self.bounds
        oxmin, oymin, oxmax, oymax = other.bounds
        return not (xmax < oxmin or xmin > oxmax or ymax < oymin or ymin > oymax)

    def compute_interaction_list(self):
        """
        Computes interaction list for the actual quadtree node 
        """
        interaction_list = []

        # if the node has a parent
        if self.parent:
            # Obtain the adjacent cells of the parent 
            parent_adjacent_cells = [child 
                                     for child in self.parent.children 
                                     if child != self]

            # Check if the children of the adjacent cells are not adjacent
            for adjacent in parent_adjacent_cells:
                for child in adjacent.children:
                    if not self.is_adjacent(child):
                        interaction_list.append(child)

        return interaction_list

    def point_in_bounds(self, point, bounds):
        x, y = point
        xmin, ymin, xmax, ymax = bounds
        return xmin <= x < xmax and ymin <= y < ymax

    def find_point_location(self, point):
        """
        Finds the leaf and parent node path for a given point 
        
        Parameters:
        -----------
        point : tuple
            Coords. (x, y) of the given point
        
        Returns:
        --------
        dict
            Dict with the keys 'leaf', 'path' and 'depth'
        """
        def _find_recursive(node, point, path):
            path.append(node)
            
            if not node.children: 
                return {
                    'leaf': node,
                    'path': path,
                    'depth': len(path) - 1
                }
            
            for child in node.children:
                if child.point_in_bounds(point, child.bounds):
                    return _find_recursive(child, point, path)
        
        return _find_recursive(self, point, path=[])    