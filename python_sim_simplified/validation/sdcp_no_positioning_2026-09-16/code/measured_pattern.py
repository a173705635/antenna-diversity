"""Measured antenna-pattern CSV loading and angular interpolation."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeasuredGrid:
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    heights: np.ndarray


def load_measured_pattern(
    path: Path,
    frequency_hz: float,
) -> dict[str, MeasuredGrid]:
    grouped: dict[str, list[tuple[float, float, complex, complex]]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if np.isclose(float(row["frequency_hz"]), frequency_hz, atol=1.0):
                grouped.setdefault(row["element_name"], []).append(
                    (
                        float(row["azimuth_deg"]) % 360.0,
                        float(row["elevation_deg"]),
                        complex(float(row["h_az_real"]), float(row["h_az_imag"])),
                        complex(float(row["h_el_real"]), float(row["h_el_imag"])),
                    )
                )
    grids = {}
    for name, samples in grouped.items():
        azimuths = np.array(sorted({sample[0] for sample in samples}))
        elevations = np.array(sorted({sample[1] for sample in samples}))
        coordinates = {
            (azimuth, elevation): (h_az, h_el)
            for azimuth, elevation, h_az, h_el in samples
        }
        heights = np.empty((elevations.size, azimuths.size, 2), dtype=complex)
        for row, elevation in enumerate(elevations):
            for column, azimuth in enumerate(azimuths):
                heights[row, column] = coordinates[(azimuth, elevation)]
        grids[name] = MeasuredGrid(azimuths, elevations, heights)
    return grids


def _bounded_neighbors(axis: np.ndarray, value: float):
    if axis.size == 1 or value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        last = axis.size - 1
        return last, last, 0.0
    upper = int(np.searchsorted(axis, value, side="right"))
    lower = upper - 1
    return lower, upper, (value - axis[lower]) / (axis[upper] - axis[lower])


def _periodic_neighbors(axis: np.ndarray, value: float):
    value %= 360.0
    upper = int(np.searchsorted(axis, value, side="right"))
    if upper == 0:
        lower, upper = axis.size - 1, 0
        low_value, high_value = axis[lower] - 360.0, axis[upper]
    elif upper == axis.size:
        lower, upper = axis.size - 1, 0
        low_value, high_value = axis[lower], axis[upper] + 360.0
    else:
        lower = upper - 1
        low_value, high_value = axis[lower], axis[upper]
    return lower, upper, (value - low_value) / (high_value - low_value)


def interpolate_measured(
    grid: MeasuredGrid,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    el0, el1, el_fraction = _bounded_neighbors(
        grid.elevations_deg, elevation_deg
    )
    az0, az1, az_fraction = _periodic_neighbors(
        grid.azimuths_deg, azimuth_deg
    )
    low = (1.0 - az_fraction) * grid.heights[el0, az0] + (
        az_fraction * grid.heights[el0, az1]
    )
    high = (1.0 - az_fraction) * grid.heights[el1, az0] + (
        az_fraction * grid.heights[el1, az1]
    )
    return (1.0 - el_fraction) * low + el_fraction * high
