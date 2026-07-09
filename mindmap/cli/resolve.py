"""
Module Name: cli/resolve.py
Description: Resolves the objects a viz target needs by discovering files on
disk rather than declaring expected entities up front.

Each requested object ("pick") is resolved independently: glob the subject's
folder for files with the right suffix, filter by whatever entities the user
pinned for that pick via `-e suffix:key=value[,key=value...]`, and narrow
interactively - one round, one prompt per remaining varying entity - until
exactly one file is left. Answering blank for an entity means "this entity
must be absent from the file", not "no preference": each pick converges to
exactly one object.

Repeating `-e` for the same suffix requests another, independently resolved,
object - that's how a target gets several objects to compare side by side
(only for suffixes a target declares as `multi`). A suffix with no `-e` at
all falls back to a single implicit pick with no entities pinned.

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
_HEMI_TOKENS = ("-lh", "-rh")  # e.g. stc's suffix is written as 'stc-lh'/'stc-rh' on disk
MAX_PROMPT_ROUNDS = 5

# Sentinel meaning "this entity must not be present on the file" - what a blank
# interactive answer resolves to (see module docstring).
_ABSENT = object()


@dataclass
class ObjectSpec:
    """Describes one object a viz target needs to load."""
    suffix: str
    multi: bool = False    # can this target accept several picks of this suffix (comparison)?
    key: str = None        # key in the returned bundle; defaults to `suffix`


def resolve_bundle(dm, subject, specs, picks_by_suffix):
    """Resolve every ObjectSpec of a target into {key: object_or_dict_of_objects}."""
    bundle = {}
    for spec in specs:
        picks = picks_by_suffix.get(spec.suffix, [])
        bundle[spec.key or spec.suffix] = resolve_object(dm, subject, spec, picks)
    return bundle


def resolve_object(dm, subject, spec, picks):
    """
    Resolve one ObjectSpec: a single loaded object, or {label: object} if spec.multi.

    `picks` is a list of entity dicts, one per requested object (see module
    docstring). An empty list means "nothing pinned - resolve the one implicit
    object", not "no objects wanted".
    """
    picks = picks or [{}]
    if len(picks) > 1 and not spec.multi:
        raise ValueError(
            f"Target doesn't support multiple '{spec.suffix}' objects (got {len(picks)})."
        )

    loader = dm.registry[spec.suffix][2]

    if not _discoverable(dm, spec.suffix):
        # Not BIDS-style (e.g. freesurfer/charm derivatives) - no discovery possible.
        if len(picks) > 1:
            raise ValueError(f"'{spec.suffix}' isn't file-discoverable; can't resolve multiple objects.")
        obj = dm.load(subject, spec.suffix, **picks[0])
        return {"obj": obj} if spec.multi else obj

    resolved = [_resolve_one(dm, subject, spec.suffix, pick) for pick in picks]
    if not spec.multi:
        return loader(resolved[0][0])

    return {
        _label(effective_filter, entities): loader(path)
        for path, entities, effective_filter in resolved
    }


def resolve_path(dm, subject, suffix, picks, multi=False):
    """
    Resolve one suffix to its filesystem path(s), without loading it.

    For targets like 'charm'/'parc' whose plot function reads the file(s)
    itself (e.g. plot_segmentation), rather than consuming a loaded object.
    Goes through the same discovery/interactive-narrowing as resolve_object.
    """
    picks = picks or [{}]
    if len(picks) > 1 and not multi:
        raise ValueError(f"'{suffix}' doesn't support multiple objects here (got {len(picks)}).")

    if not _discoverable(dm, suffix):
        path = dm.get_path(subject, suffix, **picks[0])
        return {"obj": path} if multi else path

    resolved = [_resolve_one(dm, subject, suffix, pick) for pick in picks]
    if not multi:
        return resolved[0][0]

    return {
        _label(effective_filter, entities): path
        for path, entities, effective_filter in resolved
    }


def _discoverable(dm, suffix):
    """Whether this suffix's files live under a root we can glob (raw/mm)."""
    return dm.registry[suffix][0] in ("raw", "mm")


def _resolve_one(dm, subject, suffix, filter_dict):
    """
    Narrow a suffix's candidates down to exactly one (path, entities, effective_filter),
    asking interactively - one round, every currently-varying entity at once - until
    unambiguous. `effective_filter` is `filter_dict` plus whatever got pinned along
    the way, used to build a human-readable label for the object.
    """
    candidates = [(p, e) for p, e in _discover(dm, subject, suffix) if _matches_filter(e, filter_dict)]
    if not candidates:
        raise FileNotFoundError(f"No '{suffix}' matches {_display(filter_dict)}.")

    for _ in range(MAX_PROMPT_ROUNDS):
        if len(candidates) == 1:
            path, entities = candidates[0]
            return path, entities, filter_dict

        varying = _varying_keys(candidates, filter_dict)
        if not varying:
            raise FileNotFoundError(
                f"{len(candidates)} '{suffix}' files are indistinguishable for {_display(filter_dict)}."
            )

        answers = _ask_entities(suffix, varying)
        narrowed = [(p, e) for p, e in candidates if _matches_filter(e, answers)]
        if not narrowed:
            print(f"No '{suffix}' file matches those entities. Try again.")
            continue  # keep `candidates`/`filter_dict` as-is, re-ask the same round

        candidates = narrowed
        filter_dict = {**filter_dict, **answers}

    raise FileNotFoundError(f"Gave up resolving '{suffix}' after {MAX_PROMPT_ROUNDS} attempts.")


def _discover(dm, subject, suffix):
    """List (loadable_path, parsed_entities) for every file under the subject's folder with this suffix."""
    root_key = dm.registry[suffix][0]
    root = Path(dm.roots[root_key]) / f"sub-{subject}"
    if not root.exists():
        return []

    matches = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in _SIDECAR_EXTENSIONS:
            continue
        parsed = _parse_filename(path, suffix)
        if parsed is not None:
            loadable, entities = parsed
            matches[loadable] = entities  # hemisphere pairs collapse onto the same stem path
    return list(matches.items())


def _parse_filename(path, suffix):
    """
    Parse a `sub-X[_key-value]..._suffix[-lh|-rh].ext` filename into
    (loadable_path, {entity: value}).

    Unlike pybids' parser, the suffix is matched literally (so hyphenated
    suffixes like 'i-ave' work) and entity values may themselves contain
    hyphens (so 'desc-template-coreg' keeps its full value).

    Hemisphere-paired suffixes (only 'stc' today, stored on disk as
    '..._stc-lh.stc' / '..._stc-rh.stc') resolve to the stem path without the
    hemisphere or extension - what their loader (mne.read_source_estimate)
    actually expects - so both files of a pair collapse onto one candidate.
    """
    stem = path.name.split(".", 1)[0]
    tokens = stem.split("_")
    if not tokens:
        return None

    last = tokens[-1]
    if last == suffix:
        loadable = path
    elif last in (f"{suffix}{hemi}" for hemi in _HEMI_TOKENS):
        loadable = path.parent / stem[:-len("-lh")]  # "-lh"/"-rh" are both 3 chars
    else:
        return None

    entities = {"suffix": suffix}
    for token in tokens[1:-1]:  # tokens[0] is always the leading sub-{subject}; already known
        match = _ENTITY_TOKEN.match(token)
        if match:
            entities[match.group(1)] = match.group(2)
    return loadable, entities


def _matches_filter(entities, filter_dict):
    for key, value in filter_dict.items():
        if value is _ABSENT:
            if key in entities:
                return False
        elif str(entities.get(key)) != str(value):
            return False
    return True


def _varying_keys(candidates, filter_dict):
    """Entity keys that appear among the current candidates and aren't already pinned."""
    return sorted({k for _, e in candidates for k in e} - set(filter_dict) - _NON_ENTITY_KEYS)


def _ask_entities(suffix, varying):
    """Prompt once for each varying entity; blank means 'must be absent from the file'."""
    print(f"[{suffix}] resolve by: {', '.join(varying)} (blank = must be absent)")
    answers = {}
    for key in varying:
        raw = input(f"  {key}: ").strip()
        answers[key] = raw if raw else _ABSENT
    return answers


def _label(effective_filter, entities):
    """
    Derive a short, human-readable key for one object in a multi-object bundle.

    Prefers the entities that actually drove resolution (what the user typed
    for this pick, plus anything pinned while resolving ambiguity), reading
    their values off the resolved file so the `_ABSENT` sentinel never leaks
    into a label. Falls back to the file's full entity set if resolution
    pinned nothing at all (e.g. an implicit pick that was already unambiguous).
    """
    keys = [k for k in effective_filter if k not in _NON_ENTITY_KEYS]
    parts = {k: entities[k] for k in keys if k in entities}

    if not parts:
        parts = {k: v for k, v in entities.items() if k not in _NON_ENTITY_KEYS}

    if not parts:
        return "obj"
    if len(parts) == 1:
        return str(next(iter(parts.values())))
    return "_".join(f"{k}-{v}" for k, v in sorted(parts.items()))


def _display(filter_dict):
    if not filter_dict:
        return "<no filter>"
    return ", ".join(f"{k}={'<absent>' if v is _ABSENT else v}" for k, v in filter_dict.items())


def parse_entity_flags(items):
    """
    Turn repeated `-e suffix:key=value[,key=value...]` CLI flags into
    {suffix: [pick_dict, pick_dict, ...]}.

    Each flag occurrence is one requested object ("pick"). Repeating the flag
    for the same suffix requests another, independently resolved object - the
    mechanism for side-by-side comparison (see module docstring).
    """
    by_suffix = {}
    for item in items or []:
        suffix, sep, rest = item.partition(":")
        if not sep or not rest:
            raise ValueError(f"Invalid -e '{item}': expected 'suffix:key=value[,key=value...]'.")

        pick = {}
        for pair in rest.split(","):
            key, eq, value = pair.partition("=")
            if not eq:
                raise ValueError(f"Invalid -e '{item}': expected 'key=value' in '{pair}'.")
            pick[key] = value

        by_suffix.setdefault(suffix, []).append(pick)

    return by_suffix
