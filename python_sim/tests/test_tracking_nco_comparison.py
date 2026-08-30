import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from antenna_diversity.analysis.tracking_nco import (
    CSV_COLUMNS,
    L1_CARRIER_HZ,
    compare_tracking_ncos,
    plot_tracking_nco_comparison,
    save_tracking_nco_comparison,
)
from antenna_diversity.config import SPEED_OF_LIGHT_MPS, default_config
from run_tracking_nco_comparison import build_parser, main


@pytest.fixture(scope="module")
def comparison_bundle():
    cfg = default_config()
    cfg.plot.enable = False
    original = (
        cfg.tracking.carrier_nco_mode,
        cfg.tracking.carrier_nco_clock_hz,
        cfg.tracking.carrier_nco_word_bits,
    )

    comparison = compare_tracking_ncos(
        cfg,
        clock_hz=6.6e6,
        fixed_word_bits=(30, 43),
    )

    return cfg, original, comparison


def test_comparison_uses_identical_inputs_without_mutating_config(
    comparison_bundle,
):
    cfg, original, comparison = comparison_bundle

    assert tuple(comparison.runs) == ("float", "fixed_30", "fixed_43")
    assert (
        cfg.tracking.carrier_nco_mode,
        cfg.tracking.carrier_nco_clock_hz,
        cfg.tracking.carrier_nco_word_bits,
    ) == original
    reference = comparison.runs["float"]
    for case_name in ("fixed_30", "fixed_43"):
        candidate = comparison.runs[case_name]
        assert candidate.config.random_seed == reference.config.random_seed
        np.testing.assert_array_equal(
            candidate.scene_input_v, reference.scene_input_v
        )
        np.testing.assert_array_equal(candidate.adc_codes, reference.adc_codes)
        np.testing.assert_array_equal(
            candidate.dfe_channels, reference.dfe_channels
        )
        for branch_name in cfg.diversity.modes:
            branch_key = branch_name.lower()
            candidate_acq = candidate.branches[branch_key].acquisition
            reference_acq = reference.branches[branch_key].acquisition
            assert candidate_acq["success"] == reference_acq["success"]
            if reference_acq["success"]:
                assert candidate_acq["doppler_hz"] == reference_acq[
                    "doppler_hz"
                ]
                assert candidate_acq["code_phase_samples"] == reference_acq[
                    "code_phase_samples"
                ]


def test_comparison_rows_cover_every_case_and_branch(comparison_bundle):
    cfg, _, comparison = comparison_bundle

    assert len(comparison.rows) == 3 * len(cfg.diversity.modes)
    assert {
        (row["nco_case"], row["branch"]) for row in comparison.rows
    } == {
        (case, branch.lower())
        for case in ("float", "fixed_30", "fixed_43")
        for branch in cfg.diversity.modes
    }
    assert all(tuple(row) == CSV_COLUMNS for row in comparison.rows)

    row_30 = next(
        row
        for row in comparison.rows
        if row["nco_case"] == "fixed_30" and row["branch"] == "mvdr"
    )
    expected_lsb = 6.6e6 / 2**30
    assert row_30["frequency_lsb_hz"] == pytest.approx(expected_lsb)
    assert row_30["l1_velocity_lsb_mm_s"] == pytest.approx(
        1000.0 * SPEED_OF_LIGHT_MPS / L1_CARRIER_HZ * expected_lsb
    )
    assert row_30["fcw_unique_count"] >= 1
    assert row_30["fcw_jump_count"] >= 0
    for row in comparison.rows:
        for value in row.values():
            if isinstance(value, (int, float, np.number)):
                assert np.isfinite(value)


def test_comparison_plots_stored_frequency_histories(comparison_bundle):
    cfg, _, comparison = comparison_bundle

    figures = plot_tracking_nco_comparison(comparison)
    try:
        assert tuple(figures) == (
            "tracking_nco_frequency",
            "tracking_nco_quantization_error",
        )
        assert all(
            len(figure.axes) == len(cfg.diversity.modes)
            for figure in figures.values()
        )
        mvdr_tracking = comparison.runs["fixed_30"].branches[
            "mvdr"
        ].tracking
        frequency_axis = figures["tracking_nco_frequency"].axes[2]
        frequency_line = next(
            line for line in frequency_axis.lines if line.get_label() == "fixed_30"
        )
        np.testing.assert_array_equal(
            frequency_line.get_ydata(),
            mvdr_tracking["carrier_frequency_hz"],
        )
        error_axis = figures["tracking_nco_quantization_error"].axes[2]
        error_line = next(
            line for line in error_axis.lines if line.get_label() == "fixed_30"
        )
        np.testing.assert_array_equal(
            error_line.get_ydata(),
            mvdr_tracking["carrier_frequency_quantization_error_hz"],
        )
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_comparison_saves_fixed_csv_and_nonempty_figures(
    comparison_bundle,
    tmp_path: Path,
):
    cfg, _, comparison = comparison_bundle

    paths = save_tracking_nco_comparison(comparison, tmp_path)

    assert tuple(paths) == (
        "csv",
        "frequency_figure",
        "quantization_error_figure",
    )
    assert all(path.is_absolute() for path in paths.values())
    assert all(path.parent == tmp_path.resolve() for path in paths.values())
    assert paths["frequency_figure"].stat().st_size > 0
    assert paths["quantization_error_figure"].stat().st_size > 0
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
    assert len(rows) == len(comparison.runs) * len(cfg.diversity.modes)
    assert "nan" not in paths["csv"].read_text(encoding="utf-8").lower()
    assert "inf" not in paths["csv"].read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    ("clock_hz", "word_bits"),
    [
        (0.0, (30, 43)),
        (np.nan, (30, 43)),
        (np.array(6.6e6), (30, 43)),
        (np.array([6.6e6]), (30, 43)),
        (6.6e6, ()),
        (6.6e6, (30, 30)),
        (6.6e6, (True, 43)),
        (6.6e6, (1, 43)),
    ],
)
def test_comparison_rejects_invalid_nco_sweep(clock_hz, word_bits):
    cfg = default_config()
    cfg.plot.enable = False

    with pytest.raises(ValueError, match="clock_hz|word_bits"):
        compare_tracking_ncos(
            cfg,
            clock_hz=clock_hz,
            fixed_word_bits=word_bits,
        )


def test_cli_defaults_match_l1_validation_case():
    args = build_parser().parse_args([])

    assert args.clock_hz == 6.6e6
    assert args.word_bits == (30, 43)
    assert args.output_dir == Path("results/tracking_nco")


def test_cli_parses_unique_comma_separated_word_lengths():
    parser = build_parser()

    assert parser.parse_args(["--word-bits", "24,40"]).word_bits == (24, 40)
    with pytest.raises(SystemExit):
        parser.parse_args(["--word-bits", "30,30"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--word-bits", "1,43"])


def test_cli_writes_comparison_artifacts(tmp_path: Path, capsys):
    exit_code = main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "tracking_nco_comparison.csv").stat().st_size > 0
    assert (tmp_path / "tracking_nco_frequency.png").stat().st_size > 0
    assert (
        tmp_path / "tracking_nco_quantization_error.png"
    ).stat().st_size > 0
    output = capsys.readouterr().out
    assert str(
        (tmp_path / "tracking_nco_comparison.csv").resolve()
    ) in output
