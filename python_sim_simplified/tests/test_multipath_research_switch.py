"""The research switch controls generated echoes and the dedicated runners."""

import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.config as config
import core.gnss as gnss
import core.pipeline as pipeline
from core.plotting import plot_array_response
from core.multipath.plotting import plot_multipath_results
from core.presets import default_config
from experiments.multipath import sdcp_baseline, sdcp_multipath


def test_default_disables_echo_generation_and_processing(monkeypatch):
    cfg = default_config()
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 1
    assert cfg.enable_multipath_research is False
    assert cfg.signal.multipath_paths  # The research parameters remain editable.

    def unexpected(*args, **kwargs):
        pytest.fail("disabled multipath code was called")

    monkeypatch.setattr(gnss, "_multipath_scene", unexpected)
    monkeypatch.setattr(pipeline, "detect_multipath", unexpected)
    monkeypatch.setattr(pipeline, "evaluate_multipath", unexpected)
    monkeypatch.setattr(pipeline, "combine_lcmv", unexpected)
    results = pipeline.run_end_to_end(cfg)
    assert results["truth"]["multipath_paths"] == []
    assert results["truth"]["antenna_model_diagnostics"]["multipath_paths"] == []
    assert results["multipath_detection"]["status"] == "multipath_research_disabled"
    assert results["multipath_evaluation"] is None
    assert "multipath_lcmv" not in results["branches"]
    assert plot_multipath_results(results) == {}


def test_explicit_config_and_global_default_can_enable_echoes(monkeypatch):
    cfg = default_config()
    cfg.enable_multipath_research = True
    cfg.signal.duration_ms = 2
    cfg.interferers = []
    _, truth = gnss.generate_scenario(cfg)
    assert len(truth["multipath_paths"]) == 1

    monkeypatch.setattr(config, "ENABLE_MULTIPATH_RESEARCH", True)
    assert default_config().enable_multipath_research is True


def test_disabled_spatial_plot_omits_dormant_echo_direction():
    cfg = default_config()
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 1
    results = pipeline.run_end_to_end(cfg)
    figure = plot_array_response(results)
    assert not any("Multipath" in line.get_label() for line in figure.axes[0].lines)
    plt.close(figure)


@pytest.mark.parametrize("runner", [sdcp_baseline, sdcp_multipath])
def test_disabled_runner_preserves_existing_output_and_skips_verification(
    runner, monkeypatch, tmp_path
):
    output = tmp_path / runner.OUTPUT.name
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "DURATION_MS", 1)  # Gate precedes duration preflight.

    def unexpected(*args, **kwargs):
        pytest.fail("disabled runner touched generation or verification")

    monkeypatch.setattr(runner, "reset_output", unexpected)
    monkeypatch.setattr(runner, "generate_scenario", unexpected)
    monkeypatch.setattr(runner, "verify_results" if runner is sdcp_baseline else "verify_and_plot", unexpected)
    assert runner.run_simulation() is None
    runner.main()
    assert sentinel.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("runner", [sdcp_baseline, sdcp_multipath])
def test_enabled_runner_reaches_preflight_before_output_reset(runner, monkeypatch):
    cfg = default_config()
    cfg.enable_multipath_research = True
    monkeypatch.setattr(runner, "default_config", lambda: cfg)
    monkeypatch.setattr(runner, "DURATION_MS", 1)
    with pytest.raises(ValueError):
        runner.run_simulation()
