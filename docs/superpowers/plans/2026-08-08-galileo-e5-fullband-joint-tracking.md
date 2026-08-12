# Galileo E5 Full-Band Joint Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a standalone MATLAB simulation that generates one SVID 1 Galileo E5 AltBOC(15,10) complex low-IF signal at 132 Msps and tracks all four E5 components coherently with a common DLL/FLL/PLL.

**Architecture:** Add an isolated `+galileo` package beside the existing GPS and antenna-diversity code. Generate exact ICD primary/secondary codes and the complete constant-envelope AltBOC waveform from one absolute code-time state; use the same replica engine inside a direct full-band E/P/L tracker. The standalone entry injects controlled initialization errors instead of adding acquisition, and no existing diversity or GPS algorithm is called or changed.

**Tech Stack:** MATLAB R2023a, base MATLAB only, package functions, function-based assertion tests, Galileo OS SIS ICD Issue 2.2 constants.

## Global Constraints

- Work only in `F:\antenna_diversity\matlab_sim` and documentation under `F:\antenna_diversity\docs`.
- Keep the E5 chain independent from `diversity.combine`, `gnss.generate_scenario`, `receiver.acquire`, `receiver.track`, and `run_end_to_end`.
- Use SVID 1, complex I/Q at 132 MHz, LO 1188 MHz, E5 carrier 1191.795 MHz, and low IF 3.795 MHz.
- Generate the complete E5a-I, E5a-Q, E5b-I, E5b-Q AltBOC(15,10) signal; known I-channel data symbols participate in coherent tracking.
- Do not depend on Navigation Toolbox, Communications Toolbox, Signal Processing Toolbox, or integer samples per chip.
- Use `D:\MATLAB\R2023a\bin\matlab.exe` for executable verification.
- The workspace does not expose functional Git metadata; replace commit steps with explicit test checkpoints and do not attempt commits.

---

### Task 1: E5 Configuration and ICD Codes

**Files:**
- Create: `matlab_sim/e5_default_config.m`
- Create: `matlab_sim/+galileo/e5_primary_code.m`
- Create: `matlab_sim/+galileo/e5_secondary_code.m`
- Create: `matlab_sim/tests/test_e5_codes.m`

**Interfaces:**
- Produces: `cfg = e5_default_config()`.
- Produces: `code = galileo.e5_primary_code(component, svid)` as a `1x10230` `+1/-1` row vector.
- Produces: `code = galileo.e5_secondary_code(component, svid)` as the ICD-assigned `+1/-1` row vector.
- Component identifiers are exactly `e5aI`, `e5aQ`, `e5bI`, and `e5bQ`; the first implementation accepts only `svid == 1` and rejects all other values explicitly.

- [ ] **Step 1: Write the failing ICD-code test**

Create `test_e5_codes.m` with independent ICD literals:

```matlab
function test_e5_codes()
cfg = e5_default_config();
assert(cfg.signal.fsHz == 132e6);
assert(abs(cfg.signal.ifHz - 3.795e6) < 1e-9);
assert(cfg.signal.samplesPerEpoch == 132000);

components = {'e5aI', 'e5aQ', 'e5bI', 'e5bQ'};
initialHex = {'3CEA9D', '515537', 'C5BEA1', 'E49AF0'};
secondaryHex = { ...
    '842E9', ...
    '83F6F69D8F6E15411FB8C9B1C', ...
    'E', ...
    'CFF914EE3C6126A49FD5E5C94'};
secondaryLengths = [20, 100, 4, 100];

for k = 1:numel(components)
    primary = galileo.e5_primary_code(components{k}, 1);
    assert(isequal(size(primary), [1, 10230]));
    assert(all(primary == 1 | primary == -1));
    expectedPrimary = hex_to_signal(initialHex{k}, 24);
    assert(isequal(primary(1:24), expectedPrimary));

    secondary = galileo.e5_secondary_code(components{k}, 1);
    assert(numel(secondary) == secondaryLengths(k));
    assert(isequal(secondary, ...
        hex_to_signal(secondaryHex{k}, secondaryLengths(k))));
end

assert_error(@() galileo.e5_primary_code('bad', 1), ...
    'galileo:e5_primary_code:UnknownComponent');
assert_error(@() galileo.e5_primary_code('e5aI', 2), ...
    'galileo:e5_primary_code:UnsupportedSvid');
fprintf('PASS test_e5_codes\n');
end

function signal = hex_to_signal(hexText, lengthBits)
bits = reshape(dec2bin(hex2dec(cellstr(hexText.')), 4).' - '0', 1, []);
signal = 1 - 2 * bits(1:lengthBits);
end

function assert_error(action, expectedIdentifier)
try
    action();
catch exception
    assert(strcmp(exception.identifier, expectedIdentifier));
    return;
end
error('test_e5_codes:MissingError', ...
    'Expected error %s was not raised.', expectedIdentifier);
end
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); test_e5_codes();"
```

Expected: FAIL because `e5_default_config` is undefined.

- [ ] **Step 3: Implement the standalone configuration**

`e5_default_config.m` must define these concrete fields:

```matlab
cfg.randomSeed = 81;
cfg.signal.svid = 1;
cfg.signal.carrierHz = 1191.795e6;
cfg.signal.loHz = 1188e6;
cfg.signal.ifHz = cfg.signal.carrierHz - cfg.signal.loHz;
cfg.signal.fsHz = 132e6;
cfg.signal.codeRateHz = 10.23e6;
cfg.signal.subcarrierRateHz = 15.345e6;
cfg.signal.primaryLengthChips = 10230;
cfg.signal.epochSec = 1e-3;
cfg.signal.samplesPerEpoch = round(cfg.signal.fsHz * cfg.signal.epochSec);
cfg.signal.durationMs = 30;
cfg.signal.dopplerHz = 850;
cfg.signal.initialCodePhaseChips = 1234.25;
cfg.signal.initialCarrierPhaseRad = 0.35;
cfg.signal.cn0DbHz = 52;

cfg.initialization.dopplerErrorHz = 200;
cfg.initialization.codePhaseErrorChips = 0.08;
cfg.initialization.carrierPhaseErrorRad = 0.20;

cfg.tracking.coherentMs = 1;
cfg.tracking.maxEpochs = 25;
cfg.tracking.earlyLateSpacingChips = 0.10;
cfg.tracking.fllAssistEpochs = 5;
cfg.tracking.fllGain = 0.45;
cfg.tracking.pllKp = 0.25;
cfg.tracking.pllKiHz = 8.0;
cfg.tracking.dllKpChips = 0.04;
cfg.tracking.dllKiHz = 0.20;
cfg.tracking.cn0WindowEpochs = 10;
cfg.tracking.sidePeakOffsetChips = 2/3;
cfg.tracking.sidePeakRatio = 1.10;
cfg.tracking.sidePeakConfirmEpochs = 2;

cfg.plot.enable = true;
cfg.plot.visible = 'on';
```

Validate scalar finite positive rates, `svid == 1`, integer samples per 1 ms epoch, positive duration, and `earlyLateSpacingChips < sidePeakOffsetChips`. Do not require integer samples per chip.

- [ ] **Step 4: Implement the primary-code generator**

Use these ICD constants:

| Component | Register 1 taps | Register 2 taps | Register 2 start |
|---|---:|---:|---:|
| e5aI | 40503 octal | 50661 octal | 30305 octal |
| e5aQ | 40503 octal | 50661 octal | 25652 octal |
| e5bI | 64021 octal | 51445 octal | 07220 octal |
| e5bQ | 64021 octal | 43143 octal | 03331 octal |

Decode register-start octal bits from least significant bit into cells `j=1..14`. Decode feedback bit `j` from bit `j` of the polynomial octal value, ignoring bit zero. For each chip, output `xor(register1(14), register2(14))`, then compute each modulo-2 feedback sum, insert feedback into cell 1, and shift old cells 1 through 13 into cells 2 through 14. Convert logic `0 -> +1` and `1 -> -1`.

- [ ] **Step 5: Implement the secondary-code generator**

Map SVID 1 as follows:

```matlab
e5aI -> CS201   -> 842E9
e5aQ -> CS1001  -> 83F6F69D8F6E15411FB8C9B1C
e5bI -> CS41    -> E
e5bQ -> CS10051 -> CFF914EE3C6126A49FD5E5C94
```

Expand each hexadecimal symbol most-significant bit first and apply the same logic-to-signal conversion.

- [ ] **Step 6: Run the test and verify GREEN**

Run the Step 2 command and require `PASS test_e5_codes` with exit code 0.

---

### Task 2: Complete AltBOC Replica

**Files:**
- Create: `matlab_sim/+galileo/altboc_replica.m`
- Create: `matlab_sim/tests/test_e5_altboc.m`

**Interfaces:**
- Produces: `[waveform, components] = galileo.altboc_replica(codePhaseChips, cfg, knownData)`.
- `codePhaseChips` is a finite row or column vector containing absolute continuous E5 code time; output shape matches it.
- `knownData.e5aI` contains one symbol per 20 primary epochs and `knownData.e5bI` contains one symbol per 4 primary epochs.
- `components` contains sampled `e5aI`, `e5aQ`, `e5bI`, `e5bQ`, `primaryEpoch`, and `subcarrierSubperiod` arrays.

- [ ] **Step 1: Write the failing constant-envelope test**

Create `test_e5_altboc.m`:

```matlab
function test_e5_altboc()
cfg = e5_default_config();
knownData.e5aI = [1, -1, 1];
knownData.e5bI = [1, -1, -1, 1, 1, -1, 1, 1];
phase = cfg.signal.initialCodePhaseChips + ...
    (0:cfg.signal.samplesPerEpoch-1) * cfg.signal.codeRateHz / ...
    cfg.signal.fsHz;

[s, c] = galileo.altboc_replica(phase, cfg, knownData);
assert(isequal(size(s), size(phase)));
assert(max(abs(abs(s) - 1)) < 1e-12);
assert(abs(mean(abs(s).^2) - 1) < 1e-12);
assert(all(c.subcarrierSubperiod >= 0 & c.subcarrierSubperiod <= 7));
assert(all(c.e5aI == 1 | c.e5aI == -1));
assert(all(c.e5aQ == 1 | c.e5aQ == -1));
assert(all(c.e5bI == 1 | c.e5bI == -1));
assert(all(c.e5bQ == 1 | c.e5bQ == -1));

phaseStates = mod(round(angle(s) / (pi/4)), 8);
assert(numel(unique(phaseStates)) == 8);

assert_error(@() galileo.altboc_replica([], cfg, knownData), ...
    'galileo:altboc_replica:InvalidCodePhase');
fprintf('PASS test_e5_altboc\n');
end
```

Add the same small `assert_error` local helper used by Task 1.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); test_e5_altboc();"
```

Expected: FAIL because `galileo.altboc_replica` is undefined.

- [ ] **Step 3: Implement tiered component sampling**

Inside `altboc_replica.m`, generate each component from absolute code phase:

```matlab
chip = floor(codePhaseChips);
primaryIndex = mod(chip, 10230) + 1;
primaryEpoch = floor(codePhaseChips / 10230);
secondaryIndex = mod(primaryEpoch, numel(secondary)) + 1;
```

For E5a-I, select `knownData.e5aI(floor(primaryEpoch/20)+1)`; for E5b-I, select `knownData.e5bI(floor(primaryEpoch/4)+1)`. Multiply primary, secondary, and data levels. Q components multiply primary and secondary only. Reject negative primary epochs and data vectors too short for the requested phase range.

- [ ] **Step 4: Implement the ICD 8-PSK lookup**

Use column index:

```matlab
column = 1 + 8*(e5aI > 0) + 4*(e5bI > 0) + ...
    2*(e5aQ > 0) + (e5bQ > 0);
subperiod = floor(mod(codePhaseChips * 1.5, 1) * 8);
```

Use the exact ICD table, one row per `subperiod=0..7`:

```matlab
phaseLut = [ ...
    5 4 4 3 6 3 1 2 6 5 7 2 7 8 8 1; ...
    5 4 8 3 2 3 1 2 6 5 7 6 7 4 8 1; ...
    1 4 8 7 2 3 1 2 6 5 7 6 3 4 8 5; ...
    1 8 8 7 2 3 1 6 2 5 7 6 3 4 4 5; ...
    1 8 8 7 2 7 5 6 2 1 3 6 3 4 4 5; ...
    1 8 4 7 6 7 5 6 2 1 3 2 3 8 4 5; ...
    5 8 4 3 6 7 5 6 2 1 3 2 7 8 4 1; ...
    5 4 4 3 6 7 5 2 6 1 3 2 7 8 8 1];
k = phaseLut(sub2ind(size(phaseLut), subperiod + 1, column));
waveform = exp(1j * pi/4 * k);
```

Restore the input vector orientation in every returned array.

- [ ] **Step 5: Run the test and verify GREEN**

Run the Step 2 command and require `PASS test_e5_altboc`.

---

### Task 3: Deterministic 132 Msps E5 Low-IF Scenario

**Files:**
- Create: `matlab_sim/+galileo/generate_e5_signal.m`
- Create: `matlab_sim/tests/test_e5_signal.m`

**Interfaces:**
- Produces: `[x, truth, knownData] = galileo.generate_e5_signal(cfg)`.
- `x` is one complex row vector with `round(durationMs*1e-3*fsHz)` samples.
- `truth` contains absolute code phase/rate, carrier Doppler/phase, IF, component centers, noise variance, and the noiseless baseband replica.

- [ ] **Step 1: Write the failing signal-placement test**

Create a test that disables noise and verifies real behavior:

```matlab
function test_e5_signal()
cfg = e5_default_config();
cfg.signal.durationMs = 2;
cfg.signal.cn0DbHz = inf;
[x1, truth1, data1] = galileo.generate_e5_signal(cfg);
[x2, truth2, data2] = galileo.generate_e5_signal(cfg);

assert(numel(x1) == 2 * cfg.signal.samplesPerEpoch);
assert(isequal(x1, x2));
assert(isequal(data1, data2));
assert(abs(truth1.e5aCenterHz + 11.550e6) < 1e-9);
assert(abs(truth1.e5bCenterHz - 19.140e6) < 1e-9);

z = x1 .* conj(truth1.basebandReplica);
frequencyHz = angle(sum(z(2:end) .* conj(z(1:end-1)))) * ...
    cfg.signal.fsHz / (2*pi);
assert(abs(frequencyHz - (cfg.signal.ifHz + cfg.signal.dopplerHz)) < 1e-6);
assert(all(abs(abs(x1) - 1) < 1e-12));

cfg.signal.cn0DbHz = 48;
[xn1, tn1] = galileo.generate_e5_signal(cfg);
[xn2, tn2] = galileo.generate_e5_signal(cfg);
assert(isequal(xn1, xn2));
assert(tn1.noiseVariance > 0 && tn1.noiseVariance == tn2.noiseVariance);
fprintf('PASS test_e5_signal\n');
end
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); test_e5_signal();"
```

Expected: FAIL because `galileo.generate_e5_signal` is undefined.

- [ ] **Step 3: Implement known-data generation and code time**

Reset `rng(cfg.randomSeed, 'twister')`. Generate enough deterministic `+1/-1` symbols for E5a-I at one symbol per 20 ms and E5b-I at one symbol per 4 ms. Include the initial primary epoch when calculating required vector length.

Use:

```matlab
scale = 1 + cfg.signal.dopplerHz / cfg.signal.carrierHz;
truth.codeRateHz = cfg.signal.codeRateHz * scale;
codePhase = cfg.signal.initialCodePhaseChips + ...
    (0:nSamples-1) * truth.codeRateHz / cfg.signal.fsHz;
```

- [ ] **Step 4: Implement carrier, noise, and truth output**

Generate `basebandReplica = galileo.altboc_replica(codePhase, cfg, knownData)` and:

```matlab
carrier = exp(1j * (2*pi*(cfg.signal.ifHz + ...
    cfg.signal.dopplerHz)*(0:nSamples-1)/cfg.signal.fsHz + ...
    cfg.signal.initialCarrierPhaseRad));
noiseless = basebandReplica .* carrier;

if isinf(cfg.signal.cn0DbHz)
    noiseVariance = 0;
else
    cn0Linear = 10^(cfg.signal.cn0DbHz / 10);
    noiseVariance = cfg.signal.fsHz / cn0Linear;
end
noise = sqrt(noiseVariance/2) * ...
    (randn(1, nSamples) + 1j*randn(1, nSamples));
x = noiseless + noise;
```

Set `truth.e5aCenterHz = cfg.signal.ifHz - cfg.signal.subcarrierRateHz` and `truth.e5bCenterHz = cfg.signal.ifHz + cfg.signal.subcarrierRateHz`. Keep `truth.basebandReplica` for standalone validation only; the tracker must not consume it.

- [ ] **Step 5: Run the test and verify GREEN**

Run the Step 2 command and require `PASS test_e5_signal`.

---

### Task 4: Direct Full-Band DLL/FLL/PLL

**Files:**
- Create: `matlab_sim/+galileo/track_e5_fullband.m`
- Create: `matlab_sim/tests/test_e5_tracking.m`

**Interfaces:**
- Produces: `trk = galileo.track_e5_fullband(x, fsHz, cfg, init, knownData)`.
- `init` has `dopplerHz`, `carrierPhaseRad`, and `codePhaseChips` at input sample zero.
- `trk` exposes all histories named in the approved design and does not require truth data.

- [ ] **Step 1: Write the failing high-SNR convergence test**

Create `test_e5_tracking.m`:

```matlab
function test_e5_tracking()
cfg = e5_default_config();
cfg.signal.cn0DbHz = 55;
cfg.plot.enable = false;
[x, truth, knownData] = galileo.generate_e5_signal(cfg);
init.dopplerHz = truth.dopplerHz + cfg.initialization.dopplerErrorHz;
init.carrierPhaseRad = truth.initialCarrierPhaseRad + ...
    cfg.initialization.carrierPhaseErrorRad;
init.codePhaseChips = truth.initialCodePhaseChips + ...
    cfg.initialization.codePhaseErrorChips;

trk = galileo.track_e5_fullband(x, cfg.signal.fsHz, ...
    cfg, init, knownData);
assert(trk.success);
assert(numel(trk.prompt) >= 20);
assert(all(isfinite(trk.dopplerHz)));
assert(all(isfinite(trk.codePhaseChips)));
assert(mean(abs(trk.prompt(end-9:end))) > 0.25);
assert(abs(mean(trk.dopplerHz(end-9:end)) - truth.dopplerHz) < 20);

epoch = trk.epochStartSamples(end-9:end) / cfg.signal.fsHz;
trueCode = truth.initialCodePhaseChips + epoch * truth.codeRateHz;
codeError = wrap_code(trk.codePhaseChips(end-9:end) - trueCode, 10230);
assert(abs(mean(codeError)) < 0.03);
assert(sqrt(mean(trk.pllErrorRad(end-9:end).^2)) < 0.25);
fprintf('PASS test_e5_tracking\n');
end

function value = wrap_code(value, period)
value = mod(value + period/2, period) - period/2;
end
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); test_e5_tracking();"
```

Expected: FAIL because `galileo.track_e5_fullband` is undefined.

- [ ] **Step 3: Implement validation, epoch alignment, and correlators**

Validate the input and initialization. Use exactly `round(fsHz*coherentMs*1e-3)` samples per epoch and reject noninteger epochs. For every epoch:

```matlab
sampleIndex = 0:samplesPerEpoch-1;
carrierPhase = carrierPhaseRad + 2*pi * ...
    (cfg.signal.ifHz + dopplerHz) * sampleIndex / fsHz;
wiped = block .* exp(-1j * carrierPhase);

phase = codePhaseChips + sampleIndex * codeRateHz / fsHz;
halfSpacing = cfg.tracking.earlyLateSpacingChips / 2;
earlyReplica = galileo.altboc_replica(phase + halfSpacing, cfg, knownData);
promptReplica = galileo.altboc_replica(phase, cfg, knownData);
lateReplica = galileo.altboc_replica(phase - halfSpacing, cfg, knownData);

early(epoch) = mean(wiped .* conj(earlyReplica));
prompt(epoch) = mean(wiped .* conj(promptReplica));
late(epoch) = mean(wiped .* conj(lateReplica));
```

Use absolute continuous code phase; never reduce the stored NCO phase modulo 10230.

- [ ] **Step 4: Implement discriminators and state updates**

Use:

```matlab
dllError = (abs(early)^2 - abs(late)^2) / ...
    (abs(early)^2 + abs(late)^2 + eps);
pllError = angle(prompt);
fllError = angle(prompt * conj(previousPrompt)) / (2*pi*epochSec);
```

Apply FLL only for configured assist epochs. At the start of each epoch, record the Doppler, carrier phase, code rate, and absolute code phase actually used by that epoch's local replicas; this makes `trk.codePhaseChips` align exactly with `trk.epochStartSamples`. After correlation, update Doppler with FLL plus PLL integral correction, advance continuous carrier phase with the updated Doppler and PLL proportional correction, integrate DLL frequency correction, derive code rate from carrier Doppler scaling plus DLL correction, and advance absolute code phase with DLL proportional correction. Compute sliding Prompt coherence and C/N0 using the same estimator shape as the existing tracker.

- [ ] **Step 5: Add side-peak protection with its own failing test**

Append a second case to `test_e5_tracking.m`. Use a noiseless signal, initialize code phase at `truth.initialCodePhaseChips + 2/3 + 0.02`, set `sidePeakRatio = 1.01`, and require at least one nonzero `trk.sidePeakJumpChips` followed by final wrapped code error below 0.05 chip. Run it before implementing the protection and confirm that this new assertion fails.

Then correlate monitor replicas at `phase ± sidePeakOffsetChips`. Count consecutive epochs for the stronger monitor only when its magnitude exceeds `sidePeakRatio * abs(prompt)`. On confirmation, add the signed offset once to the code phase, record the jump, and reset both counters. Do not jump when neither monitor satisfies the ratio.

- [ ] **Step 6: Run the tracking test and verify GREEN**

Run the Step 2 command and require both convergence and side-peak cases to pass.

---

### Task 5: Standalone Entry, Documentation, and Full Verification

**Files:**
- Create: `matlab_sim/run_e5_fullband_tracking.m`
- Create: `matlab_sim/tests/run_e5_tests.m`
- Modify: `matlab_sim/README.md`

**Interfaces:**
- Produces: `results = run_e5_fullband_tracking(cfg)`; omitted `cfg` calls `e5_default_config()`.
- `results` contains `config`, `truth`, `knownData`, `input`, `tracking`, and `summary`.

- [ ] **Step 1: Write the failing standalone smoke test in the E5 runner**

Create `run_e5_tests.m` so it runs tests in dependency order and then executes a short smoke scenario:

```matlab
test_e5_codes();
test_e5_altboc();
test_e5_signal();
test_e5_tracking();

cfg = e5_default_config();
cfg.plot.enable = false;
cfg.signal.cn0DbHz = 55;
results = run_e5_fullband_tracking(cfg);
assert(results.tracking.success);
assert(isfield(results, 'summary'));
assert(abs(results.summary.finalDopplerErrorHz) < 20);
fprintf('PASS all Galileo E5 full-band tests\n');
```

Run it and confirm the missing-entry failure.

- [ ] **Step 2: Implement the standalone orchestration**

The entry must:

```text
load/validate E5 configuration
-> generate one complete AltBOC complex low-IF signal
-> construct biased initial tracking state
-> execute direct full-band tracking
-> compute final Doppler/code/phase error summary from truth
-> optionally display spectrum and loop histories
-> return all numerical results
```

Keep plotting guarded by `cfg.plot.enable`; tests run with it disabled. Plot the input spectrum with E5a/E5b theoretical center markers and a second figure with Prompt magnitude, DLL error, PLL error, and Doppler history. Plotting code may remain local to this standalone entry because it has no algorithmic consumers.

- [ ] **Step 3: Document the standalone E5 workflow**

Add a Chinese README section documenting:

```matlab
cfg = e5_default_config();
cfg.plot.enable = false;
results = run_e5_fullband_tracking(cfg);
```

Document IF calculation, E5a/E5b digital centers, complete four-component known-data wipeoff, the absence of acquisition and antenna diversity, test command, major output fields, and that 66 Msps/two-separated-baseband comparison remains a later independent extension.

- [ ] **Step 4: Run all E5 tests**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); run('tests/run_e5_tests.m');"
```

Required: exit code 0 and every E5 test prints `PASS`.

- [ ] **Step 5: Run the unchanged GPS/diversity regression suite**

Run:

```powershell
& 'D:\MATLAB\R2023a\bin\matlab.exe' -batch "cd('F:/antenna_diversity/matlab_sim'); addpath(genpath(pwd)); run('tests/run_all_tests.m');"
```

Required: exit code 0 and `PASS all MATLAB simulation tests`.

- [ ] **Step 6: Run MATLAB static analysis on changed files**

Run `checkcode` on every new or modified `.m` file. Fix correctness messages, unused variables introduced by the implementation, accidental implicit expansion incompatibilities, and output-orientation issues. Re-run Steps 4 and 5 after any correction.

---

## Plan Self-Review

- Every approved signal parameter and standalone-boundary requirement maps to Tasks 1 through 5.
- The full-band tracker consumes only the composite AltBOC input and never pre-separates E5a/E5b.
- SVID 1 code constants, secondary-code assignments, AltBOC phase table, and logic-level convention are explicit and independently testable.
- Function signatures, component field names, absolute code-phase convention, and known-data layout are consistent across all tasks.
- The plan contains no dependency on antenna diversity, acquisition, proprietary toolboxes, integer samples per chip, or functional Git metadata.
- The E5 and legacy regression commands are separate so failures identify the responsible chain.
