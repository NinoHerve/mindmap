# MindMap

Mindmap is a project encompassing the entirety of the code for the Ph.D. project
entitled *"Enhancing Electrical Source Imaging with Connectomics."* It develops,
improves, validates, and evaluates new source imaging methods that use informed
priors based on connectivity research.

One limitation of source imaging is that information about source covariances must
be provided to accurately reconstruct brain activity from EEG or MEG. Some methods
attempt to compute those covariances from fMRI data measured during the same task,
but neuronal and temporal mismatches complicate this data-driven approach. This
project instead uses connectomics to infer the structural connections between
sources and incorporates new research on functional-structural information to
generate new priors, which provide indirect information about source connectivity.

Use the **Code Reference** section in the navigation to browse the package's API,
generated directly from the source's own docstrings.

## Layout

- `mindmap.core` — low-level, dependency-light primitives (stats, IO) reused
  throughout the rest of the package.
- `mindmap.anatomy` — head models, parcellations, source spaces, tractograms.
- `mindmap.analysis` — connectivity, simulation, power, and functional analyses.
- `mindmap.metric` — `MetricSuite`, for collecting, caching, and plotting a
  study's metrics and statistics.
- `mindmap.viz` — plotting.
- `mindmap.geometry`, `mindmap.io`, `mindmap.tasks`, `mindmap.cli` — supporting
  utilities and command-line entry points.

`mindmap.OLD` is legacy code, superseded and intentionally excluded from this
reference.
