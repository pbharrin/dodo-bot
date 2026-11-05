# Passive toe feasibility experiment (v6)

Branch: `experiment/passive-toe-v6`. This is **not** a replacement for v6 or its trained policies.

Generate from repository root:

```bash
python simulation/models/build_model_v6_passive_toe.py
python simulation/experiments/passive_toe/smoke_test.py
```

The v6 160 x 80 mm foot is divided into a rigid 110 mm rear sole and a 50 mm toe, hinged at the split. The toe uses a passive torsional spring (initial estimate 0.8 N m/rad), damper (0.04 N m s/rad), and 35-degree upward range. The two toe joints have **no motors**; ten existing actuators remain. Total foot mass is conserved at 150 g per side. Contact sensors from v6 monitor the rear sole only, so toe contacts require explicit diagnostics.

The smoke test checks model compilation, 10 actuators, 2 extra joints, and toe spring restoring torque. This is a structural test, **not** evidence that walking improves. The existing v6 learned policy is not assumed compatible with the changed dynamics.

Next: compare rigid v6 vs compliant v6 using the *same* controller, identical initial states, perturbations, and contact logging. Measure fall rate, toe angle, rear/toe normal forces, CoP, slip, and transition success. Sweep stiffness/damping after a baseline. CPU MuJoCo suffices for these first tests; GPU is unnecessary unless training many policies in parallel.
