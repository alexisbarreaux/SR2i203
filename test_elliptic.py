#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for elliptic testing"""
import unittest
from main import *

n = 11*17

class TestElliptic(unittest.TestCase):

    def test_elliptic_creation(self):
        a, b, x, y = create_random_elliptic(n)

        self.assertTrue(valid_point(x, y, a, b, n))

    def test_addition(self):
        a, b, x, y = create_random_elliptic(n)
        ret, l2 = elliptic_addition(n, a, b, x, y)
        if ret != 1:
            self.test_addition()
        else:
            x2, y2 = l2
            ret, l4 = elliptic_addition(n, a, b, x2, y2)
            if ret != 1:
                self.test_addition()
            else:
                x4, y4 = l4
                ret, l3 = elliptic_addition(n, a, b, x2, y2, x, y)
                if ret != 1:
                    self.test_addition()
                else:
                    x3, y3 = l3
                    _, l4_bis = elliptic_addition(n, a, b, x3, y3, x, y)
                    x4_bis, y4_bis = l4_bis
                    self.assertTrue([x4, y4] == [x4_bis, y4_bis])


if __name__ == '__main__':
    unittest.main()

