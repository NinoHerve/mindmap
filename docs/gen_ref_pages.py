"""
Generates one API-reference page per module in `mindmap/`, plus a literate
nav (`SUMMARY.md`) describing the resulting tree -- run automatically by
`mkdocs build`/`mkdocs serve` via the `mkdocs-gen-files` plugin, so the
reference section always mirrors the actual package layout instead of a
hand-maintained list that drifts out of date.

`mindmap/OLD/` (legacy, superseded code) is skipped on purpose.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "mindmap"

def _is_regular_package(py_file: Path) -> bool:
    """
    Whether every directory between `src` and `py_file` has an `__init__.py`
    -- griffe's static loader can't collect a module nested under an
    implicit namespace package (a directory missing `__init__.py`), so
    anything under one is skipped here rather than aborting the whole build.
    """
    for parent in py_file.relative_to(src).parents:
        if parent == Path("."):
            continue
        if not (src / parent / "__init__.py").exists():
            return False
    return True


skipped_namespace_packages = set()

for path in sorted(src.rglob("*.py")):
    if "OLD" in path.parts or "__pycache__" in path.parts:
        continue
    if not _is_regular_package(path):
        skipped_namespace_packages.add(path.relative_to(src).parts[0:-1])
        continue

    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if not parts:
        continue

    nav[parts[1:]] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

if skipped_namespace_packages:
    dirs = ", ".join(".".join(("mindmap",) + p) for p in sorted(skipped_namespace_packages))
    print(
        f"gen_ref_pages: skipped (missing __init__.py, implicit namespace package): {dirs}"
    )

