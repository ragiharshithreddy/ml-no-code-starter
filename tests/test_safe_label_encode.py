import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

# Delete app from sys.modules to force reload
if 'app' in sys.modules:
    del sys.modules['app']

# test_correlation globally mocks sys.modules["sklearn.preprocessing"], etc.
# We must ensure we remove those mocks.
for mod in list(sys.modules.keys()):
    if isinstance(sys.modules[mod], MagicMock):
        del sys.modules[mod]

# Import real things again
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

# Mock missing modules globally for this file
mock_modules = [
    "streamlit", "plotly", "plotly.express", "plotly.graph_objects",
    "matplotlib", "matplotlib.pyplot", "seaborn", "streamlit.components.v1",
    "requests", "ydata_profiling", "xgboost", "imblearn", "imblearn.over_sampling",
    "pycaret", "pycaret.classification", "pycaret.regression", "transformers"
]

for m in mock_modules:
    sys.modules[m] = MagicMock()

import app

class TestSafeLabelEncode(unittest.TestCase):

    def setUp(self):
        # We must re-bind real pandas and LabelEncoder to app on every test
        app.LabelEncoder = LabelEncoder
        app.pd = pd
        app.np = np

        # Reset any mocked methods we might check
        app.st.warning.reset_mock()

    def test_explicit_columns(self):
        """Test safe_label_encode with specific columns explicitly provided."""
        # Setup real DataFrame with mixed types
        df = pd.DataFrame({
            'num_col': [1, 2, 3],
            'cat_col1': ['a', 'b', 'a'],
            'cat_col2': ['yes', 'no', 'yes']
        })

        # Call the function
        result_df, encoders = app.safe_label_encode(df, columns=["cat_col1", "cat_col2"])

        # Assertions
        self.assertEqual(len(encoders), 2)
        self.assertIn("cat_col1", encoders)
        self.assertIn("cat_col2", encoders)

        self.assertEqual(df['cat_col1'].tolist(), ['a', 'b', 'a'])

        self.assertTrue(pd.api.types.is_integer_dtype(result_df['cat_col1']))
        self.assertTrue(pd.api.types.is_integer_dtype(result_df['cat_col2']))

        self.assertEqual(result_df['num_col'].tolist(), [1, 2, 3])

    def test_implicit_columns(self):
        """Test safe_label_encode with no columns provided (should detect automatically)."""
        # Setup real DataFrame with mixed types
        df = pd.DataFrame({
            'num_col': [1, 2, 3],
            'cat_col1': ['x', 'y', 'z'],
            'cat_col2': pd.Series(['a', 'b', 'c'], dtype='category')
        })

        # Call the function with columns=None
        result_df, encoders = app.safe_label_encode(df, columns=None)

        # Assertions
        self.assertEqual(len(encoders), 2)
        self.assertIn("cat_col1", encoders)
        self.assertIn("cat_col2", encoders)
        self.assertNotIn("num_col", encoders)

        self.assertEqual(df['cat_col1'].tolist(), ['x', 'y', 'z'])

        self.assertTrue(pd.api.types.is_integer_dtype(result_df['cat_col1']))
        self.assertTrue(pd.api.types.is_integer_dtype(result_df['cat_col2']))

    def test_exception_handling(self):
        """Test safe_label_encode when encoding throws an exception."""
        df = pd.DataFrame({
            'cat_col': ['a', 'b', 'c']
        })

        # Let's mock LabelEncoder just for this test
        with patch('app.LabelEncoder') as mock_label_encoder:
            # Make LabelEncoder.fit_transform raise an exception
            mock_le_instance = MagicMock()
            mock_le_instance.fit_transform.side_effect = Exception("Encoding failed")
            mock_label_encoder.return_value = mock_le_instance

            # Call the function
            result_df, encoders = app.safe_label_encode(df, columns=["cat_col"])

            # Assertions
            app.st.warning.assert_called_once()
            warning_msg = app.st.warning.call_args[0][0]
            self.assertTrue("Could not encode column 'cat_col'" in warning_msg)
            self.assertTrue("Encoding failed" in warning_msg)

            self.assertEqual(encoders, {})
            self.assertEqual(result_df['cat_col'].tolist(), ['a', 'b', 'c'])

if __name__ == '__main__':
    unittest.main()
