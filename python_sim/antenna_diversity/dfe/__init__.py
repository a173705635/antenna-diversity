"""Digital front-end processing."""

from .decimator import design_decimator
from .processing import process_channels

__all__ = ["design_decimator", "process_channels"]
