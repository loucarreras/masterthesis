from collections import deque
from typing import Iterable


def build_allowed_subtree(hierarchy, candidate_ids: Iterable[str], descendant_depth: int = 2, top_level_ids: Iterable[str] | None = None,) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    """
    Build the minimal connected sub-DAG containing `candidate_ids`, all of
    their ancestors (up to the hierarchy roots), and `descendant_depth`
    levels of descendants below each candidate.
    """
    candidate_ids = set(candidate_ids)
    allowed: set[str] = set(candidate_ids)

    # Obtain all the ancestors from all the candidates
    for cid in candidate_ids:
        allowed |= hierarchy.get_all_ancestors(cid)
    
    # Obtain the descendants up to "descendant_depth" levels lower than the candidates
    for cid in candidate_ids:
        frontier = {cid}
        for _ in range(descendant_depth):
            next_frontier = set()
            for node in frontier:
                for child in hierarchy.get_children(node):
                    allowed.add(child)
                    next_frontier.add(child)
            frontier = next_frontier
            if not frontier:
                break

    # Restrict parent/child maps to the allowed set only.
    parent_map = {n: [p for p in hierarchy.get_parents(n) if p in allowed] for n in allowed}
    child_map = {n: [c for c in hierarchy.get_children(n) if c in allowed] for n in allowed}

    return allowed, parent_map, child_map

def _topological_order(allowed: set[str], parent_map: dict[str, list[str]], child_map: dict[str, list[str]]) -> list[str]:
    """
    Return a topological ordering of the pruned subtree nodes.
    """
    in_degree = {n: len(parent_map[n]) for n in allowed}
    queue = deque(n for n in allowed if in_degree[n] == 0)
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for child in child_map.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(topo_order) < len(allowed):
        seen = set(topo_order)
        topo_order.extend(n for n in allowed if n not in seen)
    
    return topo_order

def transitive_reduction(allowed: set[str], parent_map: dict[str, list[str]], child_map: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Remove any direct edge u->v for which v is also reachable from u through
    a longer path -- i.e. keep only the most granular route between two
    connected concepts. Also drops any self-pointing edges as a safety net.
 
    This mirrors the paper's step of removing "relations that can be
    inferred by traversing G through more granular paths."
    """
    order = _topological_order(allowed, parent_map, child_map)
 
    # Full descendant set per node, computed from the ORIGINAL (unreduced)
    # child_map -- this is what makes a single pass correct: an edge u->v is
    # redundant iff v is reachable from some *other* child of u in the
    # original graph, regardless of what other edges get dropped elsewhere.
    desc: dict[str, set[str]] = {}
    for node in reversed(order):  # children before parents
        d: set[str] = set()
        for child in child_map.get(node, []):
            if child == node:
                continue  # drop self-loops
            d.add(child)
            d |= desc.get(child, set())
        desc[node] = d
 
    new_child_map: dict[str, list[str]] = {}
    for node in allowed:
        children = [c for c in child_map.get(node, []) if c != node]
        redundant = {
            c for c in children
            if any(c in desc.get(other, ()) for other in children if other != c)
        }
        new_child_map[node] = [c for c in children if c not in redundant]
 
    new_parent_map: dict[str, list[str]] = {n: [] for n in allowed}
    for node, children in new_child_map.items():
        for c in children:
            new_parent_map[c].append(node)
 
    return new_parent_map, new_child_map

def collapse_single_child_classes(allowed: set[str], parent_map: dict[str, list[str]], child_map: dict[str, list[str]], candidate_ids: set[str]) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    """
    Collapse any node that has exactly one child and is not a candidate
    concept, so that the LLM sees a more compact hierarchy.
    """
    allowed = set(allowed)
    parent_map = {n: list(parents) for n, parents in parent_map.items()}
    child_map = {n: list(children) for n, children in child_map.items()}

    changed = True
    while changed:
        changed = False
        for node in list(allowed):
            if node not in allowed or node in candidate_ids:
                continue
            children = child_map.get(node, [])
            if len(children) != 1:
                continue
            only_child = children[0]
            if only_child in candidate_ids:
                continue  # preserve the direct parent class of a candidate
 
            # Rewire: node's parents -> only_child directly, then drop node.
            parents = parent_map.get(node, [])
            for p in parents:
                p_children = child_map.setdefault(p, [])
                if node in p_children:
                    p_children.remove(node)
                if only_child not in p_children:
                    p_children.append(only_child)
 
            child_parents = parent_map.setdefault(only_child, [])
            if node in child_parents:
                child_parents.remove(node)
            for p in parents:
                if p not in child_parents:
                    child_parents.append(p)
 
            allowed.discard(node)
            parent_map.pop(node, None)
            child_map.pop(node, None)
            changed = True
 
    return allowed, parent_map, child_map

def refine_subtree(
    allowed: set[str],
    parent_map: dict[str, list[str]],
    child_map: dict[str, list[str]],
    candidate_ids: set[str],
) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    """
    Apply both refinement steps: transitive reduction
    first, then collapsing single-child class chains.
    """
    # parent_map, child_map = transitive_reduction(allowed, parent_map, child_map)
    allowed, parent_map, child_map = collapse_single_child_classes(
        allowed, parent_map, child_map, candidate_ids
    )
    return allowed, parent_map, child_map

def get_allowed_children(concept_id: str, child_map: dict[str, list[str]]) -> list[str]:
    """
    Children of `concept_id` within the pruned subtree only. Use this at
    each step of LLM navigation to show only the valid next moves.
    """
    return child_map.get(concept_id, [])


def get_roots(allowed: set[str], parent_map: dict[str, list[str]]) -> set[str]:
    """
    Concepts in the pruned subtree with no parent inside the subtree -- the
    entry points for top-down LLM navigation.
    """
    return {n for n in allowed if not parent_map.get(n)}


def render_subtree(
    hierarchy,
    parent_map: dict[str, list[str]],
    child_map: dict[str, list[str]],
    roots: set[str],
    candidate_ids: set[str],
    max_depth: int = 20,
) -> str:
    """
    Render the pruned subtree.
    """
    lines: list[str] = []

    def _walk(node: str, depth: int, seen_path: frozenset[str]):
        if depth > max_depth or node in seen_path:
            return
        label = hierarchy.get_label(node)
        marker = "*" if node in candidate_ids else " "
        lines.append(f"{'  ' * depth}{marker} {node}  {label}")
        for child in sorted(child_map.get(node, [])):
            _walk(child, depth + 1, seen_path | {node})

    for root in sorted(roots):
        _walk(root, 0, frozenset())

    return "\n".join(lines)