"""Compute the frozen long records in an isolated cache; publish complete files."""
import sys
from pathlib import Path
import single_source_twenty as campaign


subset=sys.argv[1]
primary=campaign.ROOT.resolve()
campaign.ROOT=primary/'long_prefetch'
for kind in ('drift','weak','return'):
    score,identity=campaign.evaluate(primary/'selected.py',primary/'inputs'/f'114037_{kind}_4.0',subset)
    cache=campaign.ROOT/'evaluations'/identity
    # A completed evaluation may be published; an already-started main job owns its cache.
    destination=primary/'evaluations'/identity
    if not destination.exists():
        assert destination.resolve().is_relative_to(primary)
        cache.rename(destination)
    print(kind,subset,{k:score[k] for k in ('solved','good20','error30','error50')},flush=True)
