# Planning-only feasibility search

Search in progress; no dynamic simulation is being run.

Broad sweep: RUNNING; {'endpoints_tested': 251, 'geometry_passed': 3, 'release_statics_passed': 3, 'paths_tested': 6, 'paths_passed': 0}.

Finer-waypoint trial: RUNNING; {'endpoints_tested': 41, 'geometry_passed': 1, 'release_statics_passed': 1, 'paths_tested': 1, 'paths_passed': 0}.

The initial checkpoint used an overly restrictive exact endpoint COM target. The current search uses physical support margin and COM height for release acceptance, preserving clearance, joint, contact, friction and torque requirements.

See [current.json](current.json), [refined.json](refined.json), and [search instructions](../../experiments/FEASIBILITY_SEARCH.md). Finite local-search failure does not prove model infeasibility.
