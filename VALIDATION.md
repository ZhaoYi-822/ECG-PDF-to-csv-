# v1.7 validation

- 19 automated tests passed.
- New independent synthetic cases draw known +0.8 mV and -0.5 mV triangular peaks at several widths and stroke thicknesses. Recovered amplitude errors are within 0.035 mV in those cases; a baseline check rejects blanket gain or offset changes.
- A 300-column/s synthetic trace resampled to 500 Hz retains the known positive peak within 0.035 mV.
- The original report was reconverted. The first R peak at 0.834 s is 0.382353 mV, compared with 0.243586 mV in v1.6 and an approximately 0.38-0.40 mV visual grid reading.
- The first-peak comparison and full source overlay were visually inspected. These checks support the corrected case; they do not establish accuracy for all peaks or recover original watch samples.
- Default zero lines remain 826.5, 1255.5, 1684.5 px; scale remains 119 px/mV.
- Each segment: 5000 rows. Total: 15000 rows, 0-29.998 s. Missing samples remain 171 in the first segment and zero in the other two segments.
- Fixed default grid zero references, no-click wheel zoom, English UI and documentation are retained.
- GUI source was compiled; this environment has no graphical display. Real mouse interaction and Windows EXE launch were not tested here.
- Source package only; Windows launch/build scripts are included.

- All 34 prominent connected upper source peaks were checked against their colored outer-ink tips, excluding isolated color specks. Corrected CSV peaks were at most 0.0207 mV below those tips. Outer ink is an upper bound, not ground-truth voltage.
