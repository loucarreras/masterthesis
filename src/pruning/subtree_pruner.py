"""
subtree_pruner.py

Builds a pruned sub-DAG of the SNOMED CT hierarchy (built against your
SNOMEDHierarchy class) that contains only:

  - the candidate concepts produced by upstream candidate generation
    (BM25 / dense retrieval / lexical index)
  - their ancestors, via hierarchy.get_all_ancestors(), so candidates stay
    connected to the hierarchy root(s) -- two unrelated candidates will
    meet at their nearest common ancestor
  - `descendant_depth` levels of descendants below each candidate, so an
    LLM navigating the pruned hierarchy can still refine a candidate to
    something more specific than the candidate itself

This lets the LLM-navigation step reason over a small, connected graph
instead of the full SNOMED CT hierarchy.

Uses hierarchy.get_parents / hierarchy.get_children / hierarchy.get_label /
hierarchy.get_all_ancestors directly -- no reimplementation of traversal
that SNOMEDHierarchy already does for you.
"""

from typing import Iterable


def build_allowed_subtree(
    hierarchy,
    candidate_ids: Iterable[str],
    descendant_depth: int = 2,
    top_level_ids: Iterable[str] | None = None,
) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    """
    Build the minimal connected sub-DAG containing `candidate_ids`, all of
    their ancestors (up to the hierarchy roots), and `descendant_depth`
    levels of descendants below each candidate.

    Returns:
        allowed:     set of concept_ids kept in the pruned subtree
        parent_map:  concept_id -> list of parent concept_ids, restricted to `allowed`
        child_map:   concept_id -> list of child concept_ids, restricted to `allowed`
    """
    candidate_ids = set(candidate_ids)
    allowed: set[str] = set(candidate_ids)

    # 1) Ancestors of every candidate, all the way to the root(s). This is
    #    what keeps the subtree connected.
    for cid in candidate_ids:
        allowed |= hierarchy.get_all_ancestors(cid)
    
    # 2) Descendants of every candidate, but only `descendant_depth` levels
    #    down -- so the LLM can refine, but doesn't get the whole subtree
    #    below a broad candidate.
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

    # 3) Restrict parent/child maps to the allowed set only.
    parent_map = {n: [p for p in hierarchy.get_parents(n) if p in allowed] for n in allowed}
    child_map = {n: [c for c in hierarchy.get_children(n) if c in allowed] for n in allowed}

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
    Render the pruned subtree as indented text, e.g. to drop straight into
    an LLM prompt. Candidate concepts are flagged with `*` so the LLM knows
    which nodes came from candidate generation vs. which are ancestor /
    descendant context only.

    Note: because SNOMED is a polyhierarchy, a node can legitimately appear
    under more than one parent -- `seen_path` only blocks cycles along the
    current root-to-node path, it does not deduplicate repeated appearances
    of the same node under different parents.
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