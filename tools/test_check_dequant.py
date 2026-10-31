"""Fixture-free mutations of the dequant oracle's corpus and bitwise gate."""

from __future__ import annotations

import unittest

import numpy as np

from check_dequant import GGML_NAMES, TYPES, compare_values, iq4_xs_probes


class DequantChecks(unittest.TestCase):
    def test_iq4_xs_uses_pinned_ggml_symbol_and_raw_block_geometry(self) -> None:
        self.assertEqual(TYPES["iq4_xs"], (136, 256, [0]))
        self.assertEqual(GGML_NAMES["iq4_xs"], "iq4_xs")

    def test_iq4_xs_probes_flip_every_packed_bit_independently(self) -> None:
        blocks = iq4_xs_probes()
        self.assertEqual(blocks.dtype, np.uint8)
        self.assertEqual(blocks.shape, (1 + 136 * 8 + 8 + 64, 136))
        self.assertTrue(blocks.flags.c_contiguous)
        delta = blocks[1 : 1 + 136 * 8] ^ blocks[0]
        for bit, row in enumerate(delta):
            with self.subTest(bit=bit):
                self.assertEqual(np.flatnonzero(row).tolist(), [bit // 8])
                self.assertEqual(int(row[bit // 8]), 1 << (bit % 8))
        scales = blocks[:, :2].copy().reshape(-1).view("<f2")
        self.assertTrue(np.isfinite(scales).all())
        self.assertTrue(np.any(scales.view("<u2") == 0x8000))
        self.assertTrue(np.any(scales.view("<u2") == 0x0001))
        self.assertTrue(np.any(scales.view("<u2") == 0xFBFF))

    def test_iq4_xs_scale_and_nibble_probes_cover_each_group(self) -> None:
        blocks = iq4_xs_probes()[-64:]
        high = blocks[:, 2:4].copy().reshape(-1).view("<u2")
        np.testing.assert_array_equal(blocks[:, :2], np.tile([0, 0x3E], (64, 1)))
        for group in range(8):
            low = (blocks[:, 4 + group // 2] >> (4 * (group % 2))) & 15
            code = low | (((high >> (2 * group)) & 3) << 4)
            self.assertEqual(sorted(code.tolist()), list(range(64)))
            q = blocks[:, 8 + 16 * group : 24 + 16 * group]
            np.testing.assert_array_equal(q & 15, np.tile(np.arange(16), (64, 1)))
            np.testing.assert_array_equal(q >> 4, np.tile(np.arange(15, -1, -1), (64, 1)))

    def test_bitwise_gate_rejects_every_group_nibble_and_block_boundary_mutation(self) -> None:
        want = np.linspace(-127, 113, 512, dtype=np.float32)
        self.assertEqual(compare_values(want.copy(), want, 256), (True, ""))
        positions = {0, 15, 16, 31, 255, 256, 511}
        positions.update(range(0, 512, 32))
        for pos in sorted(positions):
            with self.subTest(position=pos):
                got = want.copy()
                got.view(np.uint32)[pos] ^= np.uint32(1)
                same, detail = compare_values(got, want, 256)
                self.assertFalse(same)
                self.assertIn(f"first at {pos} (block {pos // 256})", detail)
                self.assertIn("1 of 512 values differ", detail)

    def test_bitwise_gate_rejects_signed_zero_and_missing_or_extra_blocks(self) -> None:
        want = np.zeros(512, dtype=np.float32)
        got = want.copy()
        got.view(np.uint32)[256] = 0x80000000
        self.assertFalse(compare_values(got, want, 256)[0])
        for got in (want[:256], np.zeros(768, dtype=np.float32)):
            same, detail = compare_values(got, want, 256)
            self.assertFalse(same)
            self.assertIn("shape mismatch", detail)


if __name__ == "__main__":
    unittest.main()
