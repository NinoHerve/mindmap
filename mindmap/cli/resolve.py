"""
Module Name: cli/resolve.py
Description: Resolves the objects a viz target needs by discovering files on
disk rather than declaring expected entities up front.

For each object: glob the subject's folder for files with the right suffix
(+ whatever entities were pinned for that suffix on the command line via
`-e suffix:key=value`). If there's exactly one match, load it, no questions
asked. If there are several, find which entities actually vary between the
remaining candidates and ask for those - one per entity, blank allowed (no
assumption that any given entity is present on every file) - then check a
file with that exact combination exists before continuing, re-asking the
whole round otherwise.

Filesystem globbing is used instead of the BIDSLayout index because this
project's custom derivative suffixes (ave, i-ave, con, eig, ...) aren't part
of the BIDS suffix vocabulary and pybids silently excludes them from its
index. Entities are parsed with a small local parser rather than
`layout.parse_file_entities`, which also turned out to be unreliable here:
it drops hyphenated suffixes like 'i-ave'/'v-ave' entirely, and truncates
entity values that themselves contain a hyphen (e.g. desc-template-coreg).
"""

import re
from dataclasses import dataclass
from pathlib import Path

_NON_ENTITY_KEYS = {"subject", "suffix", "extension", "datatype"}
_ENTITY_TOKEN = re.compile(r"^([a-zA-Z0-9]+)-(.+)$")
# Sidecars (LUT color tables, JSON metadata) share their suffix with the primary data file
# (e.g. sub-X_..._dseg.txt next to sub-X_..._dseg.nii.gz) but are never resolved on their
# own - loaders derive their path from the primary file's. Excluded so discovery doesn't
# treat them as an ambiguous extra candidate for the object itself.
_SIDECAR_EXTENSIONS = {".txt", ".json"}
MAX_PROMPT_ROUNDS = 5


@dataclass
class ObjectSpec:
    """Describes one object a viz target needs to load."""
    suffix: str
    multi: bool = False    # can the user select several objects (e.g. compare multiple conditions)?
    key: str = None        # key in the returned bundle; defaults to `suffix`


def resolve_bundle(dm, subject, specs, entities_by_suffix):
    """Resolve every ObjectSpec of a target into {key: object_or_dict_of_objects}."""
    bundle = {}
    for spec in specs:
        cli_filter = entities_by_suffix.get(spec.suffix, {})
        bundle[spec.key or spec.suffix] = resolve_object(dm, subject, spec, cli_filter)
    return bundle


def resolve_object(dm, subject, spec, cli_filter):
    """Resolve one ObjectSpec: a single loaded object, or {label: object} if spec.multi."""
    chosen, base_filter, discoverable = _resolve(dm, subject, spec, cli_filter)

    if not discoverable:
        # Not BIDS-style (e.g. freesurfer/charm derivatives) - no discovery possible.
        obj = dm.load(subject, spec.suffix, **base_filter)
        return {"obj": obj} if spec.multi else obj

    loader = dm.registry[spec.suffix][2]
    if not spec.multi:
        return loader(chosen[0][0])

    return {_label(entities, base_filter): loader(path) for path, entities in chosen}


def resolve_path(dm, subject, suffix, cli_filter, multi=False):
    """
    Resolve one suffix to its filesystem path(s), without loading it.

    For targets like 'charm'/'parc' whose plot function reads the file(s)
    itself (e.g. plot_segmentation), rather than consuming a loaded object.
    Goes through the same discovery/interactive-narrowing as resolve_object.
    """
    spec = ObjectSpec(suffix=suffix, multi=multi)
    chosen, base_filter, discoverable = _resolve(dm, subject, spec, cli_filter)

    if not discoverable:
        path = dm.get_path(subject, suffix, **base_filter)
        return {"obj": path} if multi else path

    if not multi:
        return chosen[0][0]

    return {_label(entities, base_filter): path for path, entities in chosen}


def _resolve(dm, subject, spec, cli_filter):
    """
    Core discovery + interactive narrowing shared by resolve_object/resolve_path.

    Returns (chosen, base_filter, discoverable). `chosen` is a list of
    (path, entities) tuples when discoverable is True; ignore it otherwise
    and rebuild the path directly from `base_filter`.
    """
    all_matches = _discover(dm, subject, spec.suffix)
    if all_matches is None:
        return None, cli_filter, False

    matches = [(p, e) for p, e in all_matches if _matches_filter(e, cli_filter)]
    if not matches:
        raise FileNotFoundError(f"No '{spec.suffix}' matches subject={subject}, filters={cli_filter}.")

    chosen = matches
    base_filter = cli_filter
    for _ in range(MAX_PROMPT_ROUNDS):
        varying = _varying_keys(chosen, cli_filter)
        if not varying or len(chosen) == 1:
            break

        answers = _ask_entities(spec.suffix, varying)
        narrowed = [(p, e) for p, e in chosen if _matches_filter(e, answers)]
        if not narrowed:
            print(f"No '{spec.suffix}' file matches those entities. Try again.")
            continue  # keep `chosen`/`base_filter` as-is, re-ask the same round

        chosen = narrowed
        base_filter = {**cli_filter, **answers}
        if spec.multi:
            break  # one round is enough; leaving an entity blank means "take every remaining value"
    else:
        raise FileNotFoundError(f"Gave up resolving '{spec.suffix}' after {MAX_PROMPT_ROUNDS} attempts.")

    return chosen, base_filter, True


def _discover(dm, subject, suffix):
    """List (path, parsed_entities) for every file under the subject's folder with this suffix."""
    root_key = dm.registry[suffix][0]
    if root_key not in ("raw", "mm"):   # Add 'fs' and 'charm'
        return None

    root = Path(dm.roots[root_key]) / f"sub-{subject}"
    if not root.exists():
        return []

    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in _SIDECAR_EXTENSIONS:
            continue
        entities = _parse_filename(path, suffix)
        if entities is not None:
            matches.append((path, entities))
    return matches


def _parse_filename(path, suffix):
    """
    Parse a `sub-X[_key-value]..._suffix.ext` filename into {entity: value}.

    Unlike pybids' parser, the suffix is matched literally (so hyphenated
    suffixes like 'i-ave' work) and entity values may themselves contain
    hyphens (so 'desc-template-coreg' keeps its full value).
    """
    stem = path.name.split(".", 1)[0]
    tokens = stem.split("_")
    if not tokens or tokens[-1] != suffix:
        return None

    entities = {"suffix": suffix}
    for token in tokens[1:-1]:  # tokens[0] is always the leading sub-{subject}; already known
        match = _ENTITY_TOKEN.match(token)
        if match:
            entities[match.group(1)] = match.group(2)
    return entities


def _matches_filter(entities, filter_dict):
    for key, value in filter_dict.items():
        allowed = value if isinstance(value, list) else [value]
        if str(entities.get(key)) not in {str(v) for v in allowed}:
            return False
    return True


def _varying_keys(matches, base_filter):
    """Entity keys that differ across the current candidates and aren't already pinned."""
    return sorted({k for _, e in matches for k in e} - set(base_filter) - _NON_ENTITY_KEYS)


def _ask_entities(suffix, varying):
    """Prompt once for each varying entity; blank means 'no filter on this entity'."""
    print(f"[{suffix}] resolve by: {', '.join(varying)} (blank = no filter)")
    answers = {}
    for key in varying:
        raw = input(f"  {key}: ").strip()
        if raw:
            answers[key] = raw
    return answers


def _label(entities, base_filter):
    """Derive a short, human-readable key for one object in a multi-object bundle."""
    extra = {k: v for k, v in entities.items() if k not in base_filter and k not in _NON_ENTITY_KEYS}
    if len(extra) == 1:
        return str(next(iter(extra.values())))
    if not extra:
        return "obj"
    return "_".join(f"{k}-{v}" for k, v in sorted(extra.items()))


def parse_entity_flags(items):
    """
    Turn repeated `-e suffix:key=value` CLI flags into {suffix: {key: value_or_[values]}}.

    Repeating the same suffix:key pins several values for that one entity
    (OR-matched); repeating with different keys pins several entities on the
    same suffix.
    """
    by_suffix = {}
    for item in items or []:
        suffix, sep, rest = item.partition(":")
        key, eq, value = rest.partition("=")
        if not sep or not eq:
            raise ValueError(f"Invalid -e '{item}': expected 'suffix:key=value'.")
        by_suffix.setdefault(suffix, {}).setdefault(key, []).append(value)

    return {
        suffix: {k: (v[0] if len(v) == 1 else v) for k, v in entities.items()}
        for suffix, entities in by_suffix.items()
    }
