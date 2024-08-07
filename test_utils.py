import unittest
from src.utils.utils import handle_outliers
import pandas as pd

class TestUtils(unittest.TestCase):
    def test_handle_outliers(self):
        data = {
            'A': [1, 2, 3, 4, 5, 100],
            'B': [1, 2, 3, 4, 5, 100]
        }
        df = pd.DataFrame(data)
        filtered_df = handle_outliers(df)
        self.assertEqual(len(filtered_df), 5)

if __name__ == '__main__':
    unittest.main()
