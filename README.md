# Dodo Bot

A bird-inspired, 3D-printed biped with the top of its torso **500 mm above the ground in its nominal standing pose**. Head/neck height is additional.

## Design baseline

- **Eight actuated degrees of freedom:** hip roll, hip pitch, knee pitch, and ankle/hock pitch on each leg.
- Eaglepower **8308, 90 KV BLDC** motors; model identified by the user from the reference BOM. Mounting dimensions and electrical/thermal limits remain to be verified.
- Rope capstan reductions targeting approximately **8:1**.
- **Makerbase MKS XDrive Mini is the preferred motor-controller candidate**, replacing the proposed S1 for cost/availability reasons. Qualify its board-specific legacy firmware, CAN behavior, encoder integration and thermal performance on one actuator first.
- Arduino **UNO Q preferred**, subject to a CAN, timing, and policy-inference bench test.
- Walking policies trained on a separate server and executed onboard.
- Printed structural links and housings with metal bearings, shafts, and fasteners.
- Editable CAD sources retained alongside STEP exports.

This is a **planning baseline, not a validated mechanical design or hardware configuration**. No motor settings, production CAD, or walking policy have been validated yet.

Read the [project plan](docs/project-plan.md) for the design sequence, component inventory, sizing examples, and acceptance gates. Read the [CAD conventions](cad/README.md) before creating parts.

## First milestone

Build and characterize **one actuator and its test fixture**, while developing a simple 500 mm torso-height kinematic layout. Measure usable travel, output torque, temperature, mass, rope settling, and repeatability before committing to eight actuators or fixed link dimensions.

## References

- [Aaed Musa Capstan-Drive](https://github.com/aaedmusa/Capstan-Drive), reviewed at commit `9c032fe83c1d9f3edaa0a6492fcbc8ce92e62ecf`.
- User-provided `dino.mp4`: visual reference for folded legs, broad feet, and body motion. The video has not been added to this public repository.
- Makerbase, Arduino, ODrive and encoder-manufacturer references are linked in the project plan.

Initial plan: 2026-10-26.
