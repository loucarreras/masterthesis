import pandas as pd
from collections import defaultdict, deque

IS_A = "116680003"
SNOMED_ROOT = "138875005"

class SNOMEDHierarchy:

    def __init__(self, concept_file, relationship_file, description_file):
        # Load SNOMED CT data from RF2 files
        self.concepts = pd.read_csv(concept_file, sep="\t", dtype=str)
        self.rels = pd.read_csv(relationship_file, sep="\t", dtype=str)
        self.descriptions = pd.read_csv(description_file, sep="\t", dtype=str, keep_default_na=False, na_filter=False)

        # Only active concepts
        self.concepts = self.concepts[self.concepts["active"] == "1"]
        self.rels = self.rels[self.rels["active"] == "1"]
        self.descriptions = self.descriptions[self.descriptions["active"] == "1"]

        # IS-A only
        self.is_a = self.rels[self.rels["typeId"] == IS_A]
        self.attr_rels = self.rels[self.rels["typeId"] != IS_A]

        # Hierarchy maps
        self.parent_map = self.build_parent_map()
        self.child_map = self.build_child_map()
        self.label_map= self.build_label_map()
        self.attribute_map = self.build_attribute_map()

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

    def build_attribute_map(self):

        attr_map = defaultdict(list)

        for _, row in self.attr_rels.iterrows():
            source = row["sourceId"]
            type_id = row["typeId"]
            destination = row["destinationId"]
            attr_map[source].append((type_id, destination))

        return dict(attr_map)
    
    def build_full_synonym_map(self):
        return self.descriptions.groupby("conceptId")["term"].apply(list).to_dict()
     
    def get_children(self, concept_id):
        return self.child_map.get(concept_id, [])
    
    def get_parents(self, concept_id):
        return self.parent_map.get(concept_id, [])

    def get_label(self, concept_id):
        return self.label_map.get(concept_id, concept_id)
    
    def get_attributes(self, concept_id):
        return self.attribute_map.get(concept_id, [])

    def get_all_descriptions(self, concept_id):
        rows = self.descriptions[self.descriptions["conceptId"] == concept_id]
        return rows["term"].tolist()
    
    def get_all_synonyms(self, concept_id):
        rows = self.descriptions[
            (self.descriptions["conceptId"] == concept_id) &
            (self.descriptions["typeId"] == "900000000000013009") # Synonym
        ]
        return rows["term"].tolist()

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
    
    # TEXT DEFINITIONS
    
    def load_text_definitions(self, text_def_file):

        df = pd.read_csv(text_def_file, sep="\t", dtype=str)

        df = df[df["active"] == "1"]

        self.text_definitions = df

        self.definition_map = (
            df.groupby("conceptId")["term"]
            .apply(list)
            .to_dict()
        )

        self.definition_count_map = {
            k: len(v) for k, v in self.definition_map.items()
        }

    def has_text_definition(self, concept_id):
        return concept_id in self.definition_map


    def get_text_definitions(self, concept_id):
        return self.definition_map.get(concept_id, [])


    def get_num_text_definitions(self, concept_id):
        return self.definition_count_map.get(concept_id, 0)


    def get_text_definition_length(self, concept_id):
        defs = self.get_text_definitions(concept_id)
        if not defs:
            return 0
        return sum(len(d.split()) for d in defs)

    def lang_refset_loader(self, lang_refset_file):

        df = pd.read_csv(lang_refset_file, sep="\t", dtype=str)

        df = df[df["active"] == "1"]

        self.lang_refset = df
    
    def get_preferred_terms(self, concept_id, lang_refset_id="900000000000509007"):
        if not hasattr(self, "lang_refset"):
            raise ValueError("Language reference set not loaded. Call lang_refset_loader first.")
        
        # Get description IDs for this concept
        description_rows = self.descriptions[self.descriptions["conceptId"] == concept_id]
        description_ids = description_rows["id"].tolist()
        
        if not description_ids:
            return []
        
        # Filter lang_refset by those description IDs + refset + preferred acceptability
        preferred_rows = self.lang_refset[
            (self.lang_refset["referencedComponentId"].isin(description_ids)) &
            (self.lang_refset["refsetId"] == lang_refset_id) &
            (self.lang_refset["acceptabilityId"] == "900000000000548007")  # Preferred
        ]
        
        if preferred_rows.empty:
            return [self.get_label(concept_id)]
        
        # Join back to descriptions to get the actual term text
        matched_ids = preferred_rows["referencedComponentId"].tolist()
        terms = description_rows[description_rows["id"].isin(matched_ids)]["term"].tolist()
        return terms


    def get_acceptable_terms(self, concept_id, lang_refset_id="900000000000509007"):
        if not hasattr(self, "lang_refset"):
            raise ValueError("Language reference set not loaded. Call lang_refset_loader first.")
        
        # Get description IDs for this concept
        description_rows = self.descriptions[self.descriptions["conceptId"] == concept_id]
        description_ids = description_rows["id"].tolist()
        
        if not description_ids:
            return []
        
        # Filter lang_refset by those description IDs + refset + acceptable acceptability
        acceptable_rows = self.lang_refset[
            (self.lang_refset["referencedComponentId"].isin(description_ids)) &
            (self.lang_refset["refsetId"] == lang_refset_id) &
            (self.lang_refset["acceptabilityId"] == "900000000000549004")  # Acceptable
        ]
        
        if acceptable_rows.empty:
            return [self.get_label(concept_id)]
        
        # Join back to descriptions to get the actual term text
        matched_ids = acceptable_rows["referencedComponentId"].tolist()
        terms = description_rows[description_rows["id"].isin(matched_ids)]["term"].tolist()
        return terms

    # Distance between concepts

    def get_ancestor_distances(self, concept_id):
        distances = {concept_id: 0}
        queue = deque([concept_id])

        while queue:
            node = queue.popleft()
            d = distances[node]
            
            for parent in self.get_parents(node):
                if parent not in distances or distances[parent] > d + 1:
                    distances[parent] = d + 1
                    queue.append(parent)
        
        return distances

    def get_descendant_distances(self, concept_id):
        distances = {concept_id: 0}
        queue = deque([concept_id]) 

        while queue:
            node = queue.popleft()
            d = distances[node]
            distances[node] = d
            
            for child in self.get_children(node):
                if child not in distances or distances[child] > d + 1:
                    distances[child] = d + 1
                    queue.append(child)
        
        return distances

    def get_common_ancestors(self, concept_id1, concept_id2):
        ancestors1 = self.get_all_ancestors(concept_id1)
        ancestors2 = self.get_all_ancestors(concept_id2)
        return ancestors1.intersection(ancestors2)
    
    def get_shortest_path_length(self, concept_id1, concept_id2):
        if concept_id1 == concept_id2:
            return 0

        ancestors1 = self.get_ancestor_distances(concept_id1)
        ancestors2 = self.get_ancestor_distances(concept_id2)

        common_ancestors = set(ancestors1.keys()).intersection(set(ancestors2.keys()))

        if not common_ancestors:
            return float("inf")

        min_distance = float("inf")
        for ancestor in common_ancestors:
            distance = ancestors1[ancestor] + ancestors2[ancestor]
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_shortest_path(self, concept_id1, concept_id2):
        if concept_id1 == concept_id2:
            return [concept_id1]

        ancestors1 = self.get_ancestor_distances(concept_id1)
        ancestors2 = self.get_ancestor_distances(concept_id2)

        common_ancestors = set(ancestors1.keys()).intersection(set(ancestors2.keys()))

        if not common_ancestors:
            return []

        min_distance = float("inf")
        closest_ancestor = None
        for ancestor in common_ancestors:
            distance = ancestors1[ancestor] + ancestors2[ancestor]
            if distance < min_distance:
                min_distance = distance
                closest_ancestor = ancestor

        # Build path from concept_id1 to closest_ancestor
        path1 = []
        current = concept_id1
        while current != closest_ancestor:
            path1.append(current)
            parents = self.get_parents(current)
            if not parents:
                break
            current = parents[0]
        path1.append(closest_ancestor)

        # Build path from closest_ancestor to concept_id2
        path2 = []
        current = concept_id2
        while current != closest_ancestor:
            path2.append(current)
            parents = self.get_parents(current)
            if not parents:
                break
            current = parents[0]
        
        # Combine paths (path1 + reversed(path2))
        full_path = path1 + list(reversed(path2))
        
        return full_path
    
    def get_lowest_common_ancestor(self, concept_id1, concept_id2):
        ancestors1 = self.get_all_ancestors(concept_id1)
        ancestors2 = self.get_all_ancestors(concept_id2)

        common_ancestors = ancestors1.intersection(ancestors2)

        if not common_ancestors:
            return None

        # Find the lowest common ancestor by depth
        lowest_common_ancestor = None
        max_depth = -1

        for ancestor in common_ancestors:
            depth = len(self.get_ancestor_path(ancestor))
            if depth > max_depth:
                max_depth = depth
                lowest_common_ancestor = ancestor

        return lowest_common_ancestor
    
    def get_lowest_common_ancestors(self, concept_id1, concept_id2):
        dist_1 = self.get_ancestor_distances(concept_id1)
        dist_2 = self.get_ancestor_distances(concept_id2)

        common = set(dist_1.keys()).intersection(set(dist_2.keys()))

        if not common:
            return []
        
        min_total = min(dist_1[c] + dist_2[c] for c in common)
        candidates = [c for c in common if dist_1[c] + dist_2[c] == min_total]

        if len(candidates) == 1:
            return candidates
        
        candidate_set = set(candidates)
        most_specific = []

        for a in candidates:
            a_ancestors = set(self.get_ancestor_distances(a).keys()) - {a}
            if not (a_ancestors.intersection(candidate_set)):
                most_specific.append(a)

        return most_specific
        
    
    def get_distance_to_lca(self, concept_id1, concept_id2):
        lcas = self.get_lowest_common_ancestors(concept_id1, concept_id2)
        if not lcas:
            return {
                "lca": [],
                "distance_1_to_lca": None,
                "distance_2_to_lca": None,
                "total_distance": None
            }
        
        dist1 = self.get_ancestor_distances(concept_id1)
        dist2 = self.get_ancestor_distances(concept_id2)

        representative = lcas[0]

        return {
            "lca": representative,
            "distance_1_to_lca": dist1[representative],
            "distance_2_to_lca": dist2[representative],
            "total_distance": dist1[representative] + dist2[representative]
        }

    def get_shortest_path_distance(snomed, concept_id1, concept_id2):
        if concept_id1 == concept_id2:
            return 0
        
        visited = {concept_id1: 0}
        queue = deque([concept_id1])

        while queue:
            node = queue.popleft()
            d = visited[node]
            
            neighbors = snomed.get_parents(node) + snomed.get_children(node)

            for neighbor in neighbors:
                if neighbor == concept_id2:
                    return d + 1
                
                if neighbor not in visited:
                    visited[neighbor] = d + 1
                    queue.append(neighbor)
        
        return None

    def wu_palmer_similarity(self, concept_id1, concept_id2, root = SNOMED_ROOT):
        lcas = self.get_lowest_common_ancestors(concept_id1, concept_id2)
        
        if not lcas:
            return None
        
        dist_1 = self.get_ancestor_distances(concept_id1)
        dist_2 = self.get_ancestor_distances(concept_id2)
        
        if root not in dist_1 or root not in dist_2:
            return None
    
        depth_1 = dist_1[root]
        depth_2 = dist_2[root]

        lca = lcas[0]
        depth_lca = depth_1 - dist_1[lca]

        denominator = depth_1 + depth_2

        if denominator == 0:
            return None
        
        return (2 * depth_lca) / denominator