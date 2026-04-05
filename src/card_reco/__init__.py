"""card_reco — Pokemon card recognition from images.

Public API re-exported from :mod:`card_reco.pipeline`.
"""

from card_reco.pipeline import (
    identify_cards,
    identify_cards_from_array,
    identify_detections,
)

__all__ = ["identify_cards", "identify_cards_from_array", "identify_detections"]
