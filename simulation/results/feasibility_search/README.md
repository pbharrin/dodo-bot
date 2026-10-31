# Planning-only feasibility search

Checkpoint: broad search and two refinements completed; release-only trajectory validation is pending. No dynamic simulation has been run.

current: SEARCH_COMPLETED_NO_FEASIBLE_PATH; {'endpoints_tested': 360, 'geometry_passed': 5, 'release_statics_passed': 5, 'paths_tested': 10, 'paths_passed': 0}

refined: SEARCH_COMPLETED_NO_FEASIBLE_PATH; {'endpoints_tested': 41, 'geometry_passed': 1, 'release_statics_passed': 1, 'paths_tested': 2, 'paths_passed': 0}

constrained_lift: SEARCH_COMPLETED_NO_FEASIBLE_PATH; {'endpoints_tested': 41, 'geometry_passed': 1, 'release_statics_passed': 1, 'paths_tested': 2, 'paths_passed': 0}

Five release poses pass geometry and static force checks. A finer reverse search connects the initial stance to a zero-load release pose. The lift connection remains unverified and fails the configured clearance/margin checks.

The first checkpoint used an overly restrictive endpoint COM target; current endpoint acceptance uses physical support margin and height. See the JSON records and [search instructions](../../experiments/FEASIBILITY_SEARCH.md).
