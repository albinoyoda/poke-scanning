"""Unit tests for the CLI module."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from card_reco.cli import main
from card_reco.models import CardRecord, MatchResult


class TestCli:
    """Tests for the card-reco CLI."""

    def test_no_command_exits(self):
        """No subcommand prints help and exits with code 1."""
        with pytest.raises(SystemExit, match="1"):
            main([])

    def test_identify_missing_image(self, capsys):
        """Referencing a non-existent image prints an error and exits."""
        with pytest.raises(SystemExit, match="1"):
            main(["identify", "/nonexistent/path/card.png"])
        captured = capsys.readouterr()
        err_out = captured.err.lower() + captured.out.lower()
        assert "not found" in err_out

    def test_identify_no_matches(self, tmp_path, capsys):
        """An image with no matches prints 'No cards detected'."""
        img_path = tmp_path / "blank.png"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        with patch("card_reco.cli.identify_cards", return_value=[]):
            main(["identify", str(img_path)])
        captured = capsys.readouterr()
        assert "no cards detected" in captured.out.lower()

    def test_identify_with_matches(self, tmp_path, capsys):
        """Matches are printed with card name and distance."""
        img_path = tmp_path / "card.png"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        card = CardRecord(
            id="xy1-1",
            name="Venusaur-EX",
            set_id="xy1",
            set_name="XY",
            number="1",
            rarity="Rare",
            image_path="v.png",
            ahash="0" * 64,
            phash="0" * 64,
            dhash="0" * 64,
            whash="0" * 64,
        )
        match = MatchResult(card=card, distance=12.5, rank=1)
        mock_results = [[match]]

        with patch("card_reco.cli.identify_cards", return_value=mock_results):
            main(["identify", str(img_path)])
        captured = capsys.readouterr()
        assert "Venusaur-EX" in captured.out
        assert "12.5" in captured.out

    def test_debug_flag_creates_writer(self, tmp_path):
        """The --debug flag causes a DebugWriter to be created."""
        img_path = tmp_path / "card.png"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        debug_dir = tmp_path / "dbg"

        with patch("card_reco.cli.identify_cards", return_value=[]) as mock_id:
            main(["identify", str(img_path), "--debug", str(debug_dir)])
            # Verify debug kwarg was passed as a DebugWriter
            call_kwargs = mock_id.call_args
            assert call_kwargs is not None
            debug_arg = call_kwargs.kwargs.get("debug") or call_kwargs[1].get("debug")
            assert debug_arg is not None

    def test_top_n_and_threshold_passed(self, tmp_path):
        """--top-n and --threshold are forwarded to identify_cards."""
        img_path = tmp_path / "card.png"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        with patch("card_reco.cli.identify_cards", return_value=[]) as mock_id:
            main(["identify", str(img_path), "--top-n", "3", "--threshold", "20.0"])
            call_kwargs = mock_id.call_args
            assert call_kwargs is not None
            assert call_kwargs.kwargs.get("top_n") == 3
            assert call_kwargs.kwargs.get("threshold") == 20.0
