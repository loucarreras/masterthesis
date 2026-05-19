import pandas as pd
from collections import defaultdict, deque

IS_A = "116680003"
SNOMED_ROOT = "138875005"

class SNOMEDHierarchy:

    def __init__(self, concept_file, relationship_file, description_file):
        # Load SNOMED CT data from RF2 files
        self.concepts = pd.read_csv(concept_file, sep="\t", dtype=str)
        self.rels = pd.read_csv(relationship_file, sep="\t", dtype=str)
        self.descriptions = pd.read_csv(description_file, sep="\t", dtype=str)

        # Only active concepts
        self.concepts = self.concepts[self.concepts["active"] == "1"]
        self.rels = self.rels[self.rels["active"] == "1"]
        self.descriptions = self.descriptions[self.descriptions["active"] == "1"]

        # IS-A only
        self.is_a = self.rels[self.rels["typeId"] == IS_A]

        # Hierarchy maps
        self.parent_map = self.build_parent_map()
        self.child_map = self.build_child_map()
        self.label_map= self.build_label_map()

    def build_parent_map(self):

        parent_map = defaultdict(list)

        for _, row in self.is_a.iterrows():
            child = row["sourceId"]
            parent = row["destinationId"]
            parent_map[child].append(parent)

        return dict(parent_map)

    def build_child_map(self):

        child_map = defaultdict(list)

        for _, row in self.is_a.iterrows():
            child = row["sourceId"]
            parent = row["destinationId"]
            child_map[parent].append(child)

        return dict(child_map)
    
    def build_label_map(self):

        descriptions = self.descriptions.copy()

        # Prefer FSN if available
        fsn = descriptions[descriptions["typeId"] == "900000000000003001"]

        label_map = {}

        for _, row in fsn.iterrows():
            label_map[row["conceptId"]] = row["term"]

        return label_map
    
    def get_children(self, concept_id):
        return self.child_map.get(concept_id, [])
    
    def get_parents(self, concept_id):
        return self.parent_map.get(concept_id, [])

    # def get_label(self, concept_id):
    #     return self.label_map.get(concept_id, concept_id)

    def get_label(self, concept_id):
        row = self.descriptions[self.descriptions["conceptId"] == concept_id]
        if len(row) == 0:
            return concept_id
        return row.iloc[0]["term"]

    def is_leaf(self, concept_id):
        return len(self.get_children(concept_id)) == 0
    
    def is_polyhierarchical(self, concept_id):
        return len(self.get_parents(concept_id)) > 1
    
    # GET PATHS

    def get_ancestor_path(self, concept_id):
        path = [concept_id]
        current = concept_id

        while True:
            parents = self.get_parents(current)

            if not parents:
                break

            parent = parents[0]
            path.append(parent)
            current = parent

        return list(reversed(path))
    
    def get_all_ancestor_paths(self, concept_id):
        
        paths = []

        def dfs(node, current_path):

            parents = self.get_parents(node)

            # Reached root
            if not parents:
                paths.append(current_path[::-1])
                return

            for parent in parents:
                dfs(parent, current_path + [parent])

        dfs(concept_id, [concept_id])

        return paths

    def get_ancestor_at_level(self, concept_id, level):
        path = self.get_ancestor_path(concept_id)
        if level >= len(path):
            return path[-1]
        return path[level]
    
    def get_depths(self, concept_id):

        paths = self.get_all_ancestor_paths(concept_id)

        depths = [len(p) - 1 for p in paths]

        return {
            "min_depth": min(depths),
            "max_depth": max(depths),
            "all_depths": depths
        }
    
    def get_all_ancestors(self, concept_id):

        visited = set()

        queue = deque([concept_id])

        while queue:

            node = queue.popleft()

            for parent in self.get_parents(node):

                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)

        return visited
    
    def get_all_descendants(self, concept_id):

        visited = set()

        queue = deque([concept_id])

        while queue:

            node = queue.popleft()

            for child in self.get_children(node):

                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        return visited