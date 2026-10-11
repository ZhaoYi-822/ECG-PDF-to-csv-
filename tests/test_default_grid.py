import unittest
import numpy as np
from ecg_tool.extractor import _second_internal_major_grid_y


class DefaultGridTests(unittest.TestCase):
    def test_second_internal_major_line_excludes_border_and_minor_lines(self):
        image=np.full((300,800,3),255,dtype=np.uint8)
        for y in range(20,261,10):
            image[y:y+2,20:780]=240
        for y in range(10,261,50):
            image[y:y+2,20:780]=205
        # The orange trace obscures most of the target line; remaining grid ink
        # still verifies its center. The zero is y=160.5, not y=210.5.
        image[160:162,220:700]=[255,130,80]
        self.assertEqual(_second_internal_major_grid_y(image,(100,280,20,780),100),160.5)

    def test_no_grid_is_an_error(self):
        with self.assertRaisesRegex(ValueError,"Cannot identify"):
            _second_internal_major_grid_y(np.full((300,800,3),255,dtype=np.uint8),(100,280,20,780),100)

    def test_inconsistent_scale_is_an_error(self):
        image=np.full((300,800,3),255,dtype=np.uint8)
        image[210:212,20:780]=205
        image[260:262,20:780]=205
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            _second_internal_major_grid_y(image,(100,280,20,780),200)
