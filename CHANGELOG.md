# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-18

### Added
- `zprofile demo`: the full validation against the simulated TDR instrument in one command.
- First release: impedance profile reconstruction from frequency-domain S-parameters.
- Loss-aware layer peeling: the reflection at each step is referred back through the accumulated
  attenuation and delay of everything the wave has already passed, so later positions are not
  systematically shallow.
- Propagation model fitted from the measured transmission, or taken from a declared NVP and loss
  coefficients when only reflection data are available.
- DC-impedance anchoring from a loop-resistance measurement or from the low-frequency limit.
- Windowing (Hann, Kaiser, Blackman-Harris) with the resolution/sidelobe trade-off stated, connector
  masking, ripple periodogram and defect detection with position and severity.
- Deviation statistics against a declared nominal impedance.
- CLI, figures, 14 tests and a 17-page technical report.
