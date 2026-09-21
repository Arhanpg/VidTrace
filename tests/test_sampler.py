"""
Tests for the adaptive frame sampler.
"""

from __future__ import annotations

import unittest

import numpy as np

from vidtrace.vision.sampler import frame_difference, frame_signature


class TestFrameSignature(unittest.TestCase):
    """Tests for frame_signature()."""

    def test_output_shape(self) -> None:
        """Signature should be 54×96 float32."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        sig = frame_signature(frame)
        assert sig.shape == (54, 96)
        assert sig.dtype == np.float32

    def test_normalized_range(self) -> None:
        """Signature values should be in [0, 1]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        sig = frame_signature(frame)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0

    def test_black_frame(self) -> None:
        """Black frame should have near-zero signature."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sig = frame_signature(frame)
        assert np.mean(sig) < 0.01

    def test_white_frame(self) -> None:
        """White frame should have near-1.0 signature."""
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        sig = frame_signature(frame)
        assert np.mean(sig) > 0.95

    def test_deterministic(self) -> None:
        """Same frame should produce the same signature."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        sig1 = frame_signature(frame)
        sig2 = frame_signature(frame.copy())
        np.testing.assert_array_almost_equal(sig1, sig2)


class TestFrameDifference(unittest.TestCase):
    """Tests for frame_difference()."""

    def test_none_returns_one(self) -> None:
        """None signature should return 1.0 (treat as scene change)."""
        sig = np.zeros((54, 96), dtype=np.float32)
        assert frame_difference(None, sig) == 1.0
        assert frame_difference(sig, None) == 1.0
        assert frame_difference(None, None) == 1.0

    def test_identical_zero(self) -> None:
        """Identical frames should have 0 difference."""
        sig = np.random.rand(54, 96).astype(np.float32)
        diff = frame_difference(sig, sig.copy())
        assert diff < 0.001

    def test_opposite_high(self) -> None:
        """Maximally different frames should have high difference."""
        black = np.zeros((54, 96), dtype=np.float32)
        white = np.ones((54, 96), dtype=np.float32)
        diff = frame_difference(black, white)
        assert diff > 0.9

    def test_symmetry(self) -> None:
        """frame_difference(a, b) == frame_difference(b, a)."""
        a = np.random.rand(54, 96).astype(np.float32) * 0.5
        b = np.random.rand(54, 96).astype(np.float32) * 0.5 + 0.3
        assert abs(frame_difference(a, b) - frame_difference(b, a)) < 0.001


if __name__ == "__main__":
    unittest.main()
