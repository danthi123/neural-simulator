"""Hierarchical concept trees (path 3) — Patterson 2007 hub-and-spoke.

Architecture:
- Each concept word has a hierarchy of category memberships
  - dog: dog -> mammal -> animal -> living_thing
  - red: red -> color -> property -> attribute
  - run: run -> motion -> action -> event

- Membership encoded as `is_a` engram tags:
  - "dog is_a mammal" -> tag 'dog_isa_mammal'
  - "mammal is_a animal" -> tag 'mammal_isa_animal'

- Queries traverse the hierarchy via tag-name pattern matching:
  - "what mammals do you know?" -> search for '*_isa_mammal'
  - "is a dog an animal?" -> 1-hop: dog->mammal->animal yields YES
  - "what kind of thing is dog?" -> traverse upward to root

Biology:
- Patterson, Nestor, Rogers 2007 "hub-and-spoke" semantic memory model
- Anterior temporal lobe (ATL) is the semantic HUB; modality-specific
  cortices are SPOKES
- Lesion in ATL -> semantic dementia with characteristic graded loss
  (abstract concepts -> specific concepts as severity increases)

Implementation approach:
- Use existing engram-tag mechanism + a special "isa" relation marker
- Queries do graph traversal on tag NAMES, not neural activations
- Could later upgrade to true neural propagation (each category has
  its own engram tag, parent-child propagation via co-firing)
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# Default hierarchy for the 60-word multi-bridge vocab.
# Format: {child: parent} -> child IS_A parent
# Top of tree has no parent.
DEFAULT_HIERARCHY = {
    # Animals (set 1 + set 2)
    "dog": "mammal",
    "cat": "mammal",
    "bird": "animal",
    "person": "mammal",
    "baby": "mammal",
    "mammal": "animal",
    "animal": "living_thing",
    "tree": "plant",
    "plant": "living_thing",
    "living_thing": "thing",

    # Objects (set 1 + set 3 + set 4 + set 5)
    "apple": "food",  # food in set 5
    "ball": "object",
    "key": "object",
    "house": "structure",
    "road": "structure",
    "structure": "object",
    "river": "water_body",
    "water_body": "natural_feature",
    "fire": "natural_feature",
    "sun": "celestial",
    "moon": "celestial",
    "celestial": "natural_feature",
    "natural_feature": "thing",
    "object": "thing",

    # Body parts (set 5)
    "hand": "body_part",
    "foot": "body_part",
    "body_part": "thing",

    # Substances
    "water": "liquid",
    "drink": "liquid",
    "liquid": "substance",
    "food": "substance",
    "substance": "thing",

    # Actions (set 1 + set 2 + set 3 + set 4 + set 5)
    "go": "motion",
    "come": "motion",
    "walk": "motion",
    "run": "motion",
    "push": "motion",
    "pull": "motion",
    "motion": "action",
    "stop": "action",
    "look": "perception",
    "listen": "perception",
    "perception": "action",
    "eat": "consumption",
    "drink": "consumption",  # NOTE: same word as noun above; could disambiguate
    "consumption": "action",
    "sleep": "state_change",
    "open": "manipulation",
    "close": "manipulation",
    "give": "manipulation",
    "take": "manipulation",
    "find": "manipulation",
    "lose": "manipulation",
    "manipulation": "action",
    "speak": "communication",
    "read": "communication",
    "write": "communication",
    "communication": "action",
    "state_change": "action",
    "action": "event",

    # Properties / adjectives
    "big": "size",
    "small": "size",
    "tall": "size",
    "short": "size",
    "size": "property",
    "hot": "temperature",
    "cold": "temperature",
    "temperature": "property",
    "wet": "moisture",
    "dry": "moisture",
    "moisture": "property",
    "happy": "emotion",
    "sad": "emotion",
    "emotion": "state",
    "fast": "speed",
    "slow": "speed",
    "speed": "property",
    "full": "quantity",
    "empty": "quantity",
    "quantity": "property",
    "red": "color",
    "blue": "color",
    "color": "property",
    "new": "age",
    "old": "age",
    "age": "property",
    "clean": "condition",
    "hard": "condition",  # NOTE: also a property of solidity
    "condition": "property",
    "property": "attribute",
    "state": "attribute",

    # Directions (motors)
    "north": "direction",
    "east": "direction",
    "south": "direction",
    "west": "direction",
    "direction": "property",
}


def get_ancestors(concept: str, hierarchy: Dict[str, str] = None) -> List[str]:
    """Return all ancestors of `concept`, root-most last.

    Example: get_ancestors('dog') -> ['mammal', 'animal', 'living_thing', 'thing']
    """
    if hierarchy is None:
        hierarchy = DEFAULT_HIERARCHY
    out = []
    visited = set()
    current = concept
    while current in hierarchy:
        parent = hierarchy[current]
        if parent in visited:
            break  # cycle guard
        visited.add(parent)
        out.append(parent)
        current = parent
    return out


def get_descendants(category: str,
                     hierarchy: Dict[str, str] = None) -> List[str]:
    """Return all descendants of `category` (concepts under it).

    Example: get_descendants('mammal') -> ['dog', 'cat', 'person', 'baby']
    get_descendants('animal') -> ['mammal', 'bird', 'dog', 'cat', 'person', 'baby']
    """
    if hierarchy is None:
        hierarchy = DEFAULT_HIERARCHY
    # Build reverse map: parent -> list of children
    reverse = defaultdict(list)
    for child, parent in hierarchy.items():
        reverse[parent].append(child)
    # BFS
    out = []
    queue = list(reverse.get(category, []))
    visited = set(queue)
    while queue:
        node = queue.pop(0)
        out.append(node)
        for child in reverse.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return out


def is_a(concept: str, category: str,
          hierarchy: Dict[str, str] = None) -> bool:
    """True if concept inherits from category (any level)."""
    if concept == category:
        return True
    return category in get_ancestors(concept, hierarchy)


def common_ancestor(a: str, b: str,
                     hierarchy: Dict[str, str] = None) -> str:
    """Return the nearest common ancestor of a and b, or empty string."""
    if hierarchy is None:
        hierarchy = DEFAULT_HIERARCHY
    a_chain = [a] + get_ancestors(a, hierarchy)
    b_chain_set = set([b] + get_ancestors(b, hierarchy))
    for node in a_chain:
        if node in b_chain_set:
            return node
    return ""


def category_summary(hierarchy: Dict[str, str] = None) -> dict:
    """Return statistics: depth distribution, branching factor."""
    if hierarchy is None:
        hierarchy = DEFAULT_HIERARCHY
    depths = []
    for concept in hierarchy:
        depths.append(len(get_ancestors(concept, hierarchy)))
    # Branching factor (children per parent)
    from collections import Counter
    parent_counts = Counter(hierarchy.values())
    branching = list(parent_counts.values())
    # Top-level categories (those that are parents but never children)
    parents = set(hierarchy.values())
    children = set(hierarchy.keys())
    roots = parents - children
    return {
        "n_concepts": len(hierarchy),
        "n_roots": len(roots),
        "roots": sorted(roots),
        "max_depth": max(depths) if depths else 0,
        "mean_depth": sum(depths) / len(depths) if depths else 0,
        "max_branching": max(branching) if branching else 0,
        "mean_branching": sum(branching) / len(branching) if branching else 0,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--summary", action="store_true")
    p.add_argument("--ancestors", type=str, default=None)
    p.add_argument("--descendants", type=str, default=None)
    p.add_argument("--isa", type=str, default=None,
                    help="format: 'concept:category'")
    p.add_argument("--common", type=str, default=None,
                    help="format: 'a:b'")
    args = p.parse_args()

    if args.summary:
        s = category_summary()
        print(f"Hierarchy summary:")
        for k, v in s.items():
            print(f"  {k}: {v}")
    if args.ancestors:
        print(f"ancestors of {args.ancestors}: {get_ancestors(args.ancestors)}")
    if args.descendants:
        print(f"descendants of {args.descendants}: {get_descendants(args.descendants)}")
    if args.isa:
        c, cat = args.isa.split(":")
        print(f"is_a('{c}', '{cat}'): {is_a(c, cat)}")
    if args.common:
        a, b = args.common.split(":")
        print(f"common_ancestor('{a}', '{b}'): {common_ancestor(a, b)}")
