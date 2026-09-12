#!/usr/bin/env python3
"""Tests for the Multiplication Table Revisited assignment."""

import unittest
from unittest.mock import patch

import numpy as np

from src.multiplication_table_revisited import multiplication_table


class TestMultiplicationTable(unittest.TestCase):
    """multiplication_table(n) -> n x n array where entry (i, j) is i * j."""

    def test_content(self):
        for n in range(1, 10):
            a = multiplication_table(n)
            self.assertEqual(
                a.shape,
                (n, n),
                msg="multiplication_table(%d) should return an array of "
                "shape (%d, %d), got %r." % (n, n, n, a.shape),
            )
            for (i, j), x in np.ndenumerate(a):
                self.assertEqual(
                    i * j,
                    x,
                    msg="multiplication_table(%d) entry at position (%d, %d) "
                    "should be %d (i*j), got %r." % (n, i, j, i * j, x),
                )

    def test_uses_np_arange(self):
        with patch("numpy.arange", wraps=np.arange) as parange:
            multiplication_table(4)
            self.assertTrue(
                parange.called,
                msg="multiplication_table(4) should build its rows/columns "
                "with np.arange rather than hardcoding the values.",
            )


if __name__ == "__main__":
    unittest.main()
