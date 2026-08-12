# MATLAB RF→DFE→TRK Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a MATLAB R2023a dual-antenna GPS L1 C/A simulation from complex-equivalent RF input through ADC, DFE, diversity combining, acquisition, and closed-loop tracking.

**Architecture:** Generate one desired satellite and one CW interferer as separate spatial fields, pass each antenna channel through a repeatable RF/ADC model, process all channels through a phase-coherent DFE, and compare single-antenna, EGC, and MVDR outputs with the same acquisition/tracking receiver. Algorithms live in small MATLAB package functions; `run_end_to_end.m` only orchestrates the chain.

**Tech Stack:** MATLAB R2023a, base MATLAB functions only, package folders (`+gnss`, `+rf`, and similar), script-based assertions for tests.

## Global Constraints

- Work only in `F:\antenna_diversity\matlab_sim` and documentation under `F:\antenna_diversity\docs`.
- Treat `F:\GNSS\GNSS_Knowledge\09-接收机与工程实现` as read-only reference material.
- Use complex-equivalent IF at 4.092 MHz, RF sampling at 16.368 MHz, and DFE output at 4.092 MHz.
- Use the convention `X = antennas × samples` and `y = w' * X` everywhere.
- Do not require Signal Processing Toolbox.
- Keep random results repeatable through `cfg.randomSeed`.
- The workspace is not a Git repository; replace commit steps with explicit test checkpoints.

---

### Task 1: Configuration, C/A Code, and Array Geometry

**Files:**
- Create: `matlab_sim/default_config.m`
- Create: `matlab_sim/+gnss/ca_code.m`
- Create: `matlab_sim/+arraysim/steering_vector.m`
- Create: `matlab_sim/tests/test_core.m`

**Interfaces:**
- Produces: `cfg = default_config()`.
- Produces: `code = gnss.ca_code(prn)` as a `1×1023` double row vector containing only `-1` and `+1`.
- Produces: `a = arraysim.steering_vector(positionsM, angleDeg, wavelengthM)` as an `N×1` complex column vector.

- [ ] **Step 1: Write the core assertions**

`test_core.m` must assert:

```matlab
cfg = default_config();
code = gnss.ca_code(1);
assert(isequal(size(code), [1, 1023]));
assert(all(code == 1 | code == -1));
r = ifft(abs(fft(code)).^2);
assert(abs(real(r(1)) - 1023) < 1e-9);
assert(isequal(sort(unique(round(real(r(2:end)))).'), [-65; -1; 63]));

a = arraysim.steering_vector([0; cfg.array.spacingM], 30, cfg.signal.wavelengthM);
expected = exp(-1j * 2*pi*cfg.array.spacingM*sind(30)/cfg.signal.wavelengthM);
assert(abs(a(2) - expected) < 1e-12);
```

- [ ] **Step 2: Run the test and confirm missing-function failure**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); tests.test_core"
```

Expected: MATLAB reports `default_config` or a package function as undefined.

- [ ] **Step 3: Implement configuration and core functions**

`default_config.m` must define the approved PRN, RF/DFE rates, 40 ms duration, half-wavelength spacing, desired/interference powers and angles, RF stages, ADC properties, acquisition bins, and loop gains. Validate the derived integer samples per code period and output samples per chip.

`ca_code.m` must implement the PRN 1–32 G1/G2 tap table and reject values outside this range.

`steering_vector.m` must implement:

```matlab
a = exp(-1j * 2*pi * positionsM(:) * sind(angleDeg) / wavelengthM);
```

- [ ] **Step 4: Run `test_core.m` and require exit code 0**

---

### Task 2: Spatial Scenario and RF/ADC Model

**Files:**
- Create: `matlab_sim/+gnss/generate_scenario.m`
- Create: `matlab_sim/+rf/cascade_metrics.m`
- Create: `matlab_sim/+rf/process_frontend.m`
- Create: `matlab_sim/tests/test_rf.m`

**Interfaces:**
- Consumes: `cfg`, `gnss.ca_code`, and `arraysim.steering_vector`.
- Produces: `[xAntennaV, truth] = gnss.generate_scenario(cfg)` with `xAntennaV` sized `Nant×Nsamp`.
- Produces: `metrics = rf.cascade_metrics(stages)` where stages contain `gainDb` and `noiseFigureDb`.
- Produces: `[adcCodes, info] = rf.process_frontend(xAntennaV, cfg.rf, cfg.randomSeed)`.

- [ ] **Step 1: Write scenario/RF tests**

The test must disable noise and verify the desired-only channel ratio equals the configured effective steering vector. It must also verify Friis cascade metrics for a single stage and check that ADC codes are integer, bounded, dimension-preserving, and repeatable for the same seed.

```matlab
cfg = default_config();
cfg.jammer.enable = false;
[x, truth] = gnss.generate_scenario(cfg);
ratio = mean(x(2,:) ./ x(1,:));
assert(abs(ratio - truth.desiredSteeringEffective(2)) < 1e-10);

m = rf.cascade_metrics(struct('gainDb', 20, 'noiseFigureDb', 2));
assert(abs(m.totalGainDb - 20) < 1e-12);
assert(abs(m.noiseFigureDb - 2) < 1e-12);
```

- [ ] **Step 2: Run `test_rf.m` and confirm the missing-function failure**

- [ ] **Step 3: Implement signal generation**

Generate received code using:

```matlab
receivedCodeRate = codeRate * (1 + dopplerHz/carrierHz);
codePhase = mod((0:nSamples-1) * receivedCodeRate/fsRf - codeDelayChips, 1023);
sampledCode = code(floor(codePhase) + 1);
```

Scale desired and jammer complex envelopes using `sqrt(Pwatt * R)` and apply their steering vectors before summation. Apply fixed per-channel amplitude/phase mismatch to external fields and return nominal/effective steering, true Doppler, code delay, source components, and time vector.

- [ ] **Step 4: Implement RF cascade and ADC**

Use Friis noise factor, generate complex noise with `E{|n|^2}=k*T*B*F*R`, apply total voltage gain, optional component-wise `tanh` limiting, and signed uniform I/Q quantization. Return total gain/NF, input/output RMS, LSB, and saturation fraction per channel.

- [ ] **Step 5: Run `test_rf.m` and require exit code 0**

---

### Task 3: Phase-Coherent DFE

**Files:**
- Create: `matlab_sim/+dfe/design_decimator.m`
- Create: `matlab_sim/+dfe/process_channels.m`
- Create: `matlab_sim/tests/test_dfe.m`

**Interfaces:**
- Produces: `[h, info] = dfe.design_decimator(fsInHz, decimation, cutoffHz, numTaps)`.
- Produces: `[xOut, info] = dfe.process_channels(adcCodes, cfg.dfe)`.

- [ ] **Step 1: Write a DFE tone test**

Create two identical channels containing a 4.092 MHz complex tone. Require output rate 4.092 MHz, output length equal to `floor((N-groupDelay-1)/4)+1`, channel equality within numerical precision, and residual frequency below one FFT bin after DDC.

- [ ] **Step 2: Run `test_dfe.m` and confirm missing-function failure**

- [ ] **Step 3: Implement the toolbox-free FIR**

Use an odd number of taps, a manually evaluated normalized sinc, and a Hamming window:

```matlab
m = -(numTaps-1)/2:(numTaps-1)/2;
u = 2*cutoffHz/fsInHz * m;
s = ones(size(u));
nz = u ~= 0;
s(nz) = sin(pi*u(nz)) ./ (pi*u(nz));
h = (2*cutoffHz/fsInHz) * s;
w = 0.54 - 0.46*cos(2*pi*(0:numTaps-1)/(numTaps-1));
h = h .* w;
h = h / sum(h);
```

- [ ] **Step 4: Implement shared-NCO DDC, filtering, delay compensation, decimation, and optional output quantization**

Use the same NCO vector and FIR for all rows. Compensate integer FIR group delay before selecting every fourth sample. When output quantization is enabled, divide by the configured binary scale, round, and saturate real and imaginary parts independently.

- [ ] **Step 5: Run `test_dfe.m` and require exit code 0**

---

### Task 4: Single, EGC, and MVDR Combining

**Files:**
- Create: `matlab_sim/+diversity/combine.m`
- Create: `matlab_sim/tests/test_diversity.m`

**Interfaces:**
- Produces: `[y, info] = diversity.combine(X, desiredSteering, mode, cfg)`.
- `info` contains `weights`, `desiredResponse`, `weightNorm`, `conditionNumber`, and loaded covariance.

- [ ] **Step 1: Write deterministic combining tests**

Build an ideal two-element target and a stronger interferer at a different angle. Require target response error below `1e-10` for EGC/MVDR and require MVDR interferer response below single-channel response.

- [ ] **Step 2: Run the test and confirm missing-function failure**

- [ ] **Step 3: Implement the three modes**

Use:

```matlab
R = (X * X') / size(X, 2);
delta = cfg.diagonalLoadingFactor * trace(R) / size(R, 1);
z = (R + delta * eye(size(R))) \ desiredSteering;
w = z / (desiredSteering' * z);
y = w' * X;
```

EGC uses `w = a/(a'*a)` and single mode uses the first basis vector. Reject unknown modes and dimension mismatches.

- [ ] **Step 4: Run `test_diversity.m` and require exit code 0**

---

### Task 5: FFT Acquisition

**Files:**
- Create: `matlab_sim/+receiver/acquire.m`
- Create: `matlab_sim/tests/test_acquisition.m`

**Interfaces:**
- Produces: `acq = receiver.acquire(x, fsHz, prn, cfg.acquisition)`.
- `acq` contains `success`, `dopplerHz`, `codePhaseSamples`, `peak`, `secondPeak`, `metric`, `dopplerBins`, and `searchPower`.

- [ ] **Step 1: Write a high-SNR acquisition test**

Generate at least 4 ms of direct baseband PRN 1 with a known integer-sample code delay and 1500 Hz Doppler. Require Doppler error no larger than one configured bin and circular code-phase error no larger than one sample.

- [ ] **Step 2: Run the test and confirm missing-function failure**

- [ ] **Step 3: Implement noncoherent FFT acquisition**

For each Doppler bin and millisecond, wipe carrier, compute circular correlation with `ifft(fft(block).*conj(fft(localCode)))`, and accumulate squared magnitudes. Search the second peak only after masking a circular interval of one chip around the primary peak. Set `success = metric >= threshold`.

- [ ] **Step 4: Run `test_acquisition.m` and require exit code 0**

---

### Task 6: Closed-Loop DLL/FLL/PLL Tracking

**Files:**
- Create: `matlab_sim/+receiver/track.m`
- Create: `matlab_sim/tests/test_tracking.m`

**Interfaces:**
- Produces: `trk = receiver.track(x, fsHz, prn, acq, cfg.tracking, carrierHz, codeRateHz)`.
- `trk` contains status, epoch indices, E/P/L, DLL/PLL/FLL errors, carrier frequency, code rate, prompt magnitude, phase-lock metric, and C/N0 estimate.

- [ ] **Step 1: Write a high-SNR tracking test**

Reuse a direct baseband signal with known Doppler and code delay. Initialize from acquisition, run at least 20 epochs, and assert finite loop states, nonzero Prompt power, final Doppler within the configured tolerance, and bounded DLL/PLL RMS over the last ten epochs.

- [ ] **Step 2: Run the test and confirm missing-function failure**

- [ ] **Step 3: Implement epoch alignment and correlators**

Move the first tracking epoch to the first acquired code boundary after the acquisition interval. Generate continuous carrier/code NCO phase, 0.5-chip Early/Late replicas, and 1 ms coherent E/P/L sums.

- [ ] **Step 4: Implement discriminators and loops**

Use normalized Early-Late envelope DLL, Prompt `atan2` PLL, and adjacent-Prompt FLL. Apply FLL aid only during the configured initial epochs. Update carrier frequency/phase and code rate/phase with explicit proportional/integral gains from the configuration. Record a sliding coherent/noncoherent C/N0 approximation and phase lock metric.

- [ ] **Step 5: Run `test_tracking.m` and require exit code 0**

---

### Task 7: End-to-End Orchestration, Metrics, Plots, and Documentation

**Files:**
- Create: `matlab_sim/+analysis/compute_metrics.m`
- Create: `matlab_sim/+analysis/plot_results.m`
- Create: `matlab_sim/run_end_to_end.m`
- Create: `matlab_sim/tests/test_end_to_end.m`
- Create: `matlab_sim/tests/run_all_tests.m`
- Create: `matlab_sim/README.md`

**Interfaces:**
- Produces: `results = run_end_to_end(cfg)`; omitted `cfg` uses `default_config()`.
- Produces: `metrics = analysis.compute_metrics(results, truth)`.
- Produces: `analysis.plot_results(results)` without modifying numerical results.

- [ ] **Step 1: Write the end-to-end smoke test and aggregate test runner**

The smoke test disables plotting, runs the complete default chain, checks all expected result fields, requires the MVDR acquisition to succeed, requires nonempty finite tracking arrays, and requires MVDR jammer response below single-channel response.

`run_all_tests.m` must call each named test in order, print a pass line after each, and throw on the first failed assertion so MATLAB `-batch` returns a nonzero exit status.

- [ ] **Step 2: Run the aggregate tests and confirm missing integration failure**

- [ ] **Step 3: Implement metrics and orchestration**

The main function must:

```text
load/validate config
→ generate spatial scene
→ process RF/ADC
→ process DFE
→ loop over single, egc, mvdr
→ acquire each combined signal
→ track only successful acquisitions
→ compute comparison metrics
→ optionally plot
```

No algorithm implementation belongs in the main function.

- [ ] **Step 4: Implement diagnostic plots**

Use base MATLAB plotting for representative spectra, array responses, acquisition maps, tracking states, and a summary comparison. Plotting must be bypassed completely when `cfg.plot.enable` is false.

- [ ] **Step 5: Write the MATLAB README**

Document the modeling boundary, directory layout, one-command run instructions, test command, output fields, parameter sweep starting points, and the distinction between antenna diversity combining and interference-nulling beamforming.

- [ ] **Step 6: Run static MATLAB checks**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); files=dir('**/*.m'); for k=1:numel(files), checkcode(fullfile(files(k).folder,files(k).name),'-id'); end"
```

Review every reported message and fix correctness issues introduced by this work.

- [ ] **Step 7: Run the full verification**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); run('tests/run_all_tests.m'); cfg=default_config(); cfg.plot.enable=false; results=run_end_to_end(cfg); disp(results.metrics.summaryTable);"
```

Required result: MATLAB exits with code 0, all named tests print `PASS`, and the end-to-end metrics table is displayed.

---

## Plan Self-Review

- Every requirement in the approved design maps to Tasks 1–7.
- Package names and function signatures are consistent across producers and consumers.
- The plan contains no dependency on Signal Processing Toolbox.
- No task modifies the read-only reference directory.
- Multi-satellite, multipath, navigation decoding, and positioning remain explicitly out of scope.
- Because this workspace has no Git metadata, the plan uses executable checkpoints instead of impossible commit steps.
