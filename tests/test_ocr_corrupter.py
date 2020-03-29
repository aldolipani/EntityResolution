import unittest

from corruptor_models.ocr_corrupter import OCRCorrupter
from entity_generator import DateGenerator


class TestOCRCorrupterCase(unittest.TestCase):

    def test_date_corruption(self):
        ocr_corruptor = OCRCorrupter()
        date_generator = DateGenerator()
        num_corruptions = 0
        for n in range(1000):
            date_generator.generate()
            date = date_generator.variant()
            corrupted_date = ocr_corruptor.corrupt(date)
            if date == corrupted_date:
                num_corruptions += 1
                #print(date, corrupted_date)
        self.assertGreater(num_corruptions, 0)

if __name__ == '__main__':
    unittest.main()
