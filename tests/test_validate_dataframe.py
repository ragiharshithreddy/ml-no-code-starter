import unittest
from unittest.mock import MagicMock
import sys

# Mock all dependencies to allow importing app.py without the actual packages
mock_pd = MagicMock()
mock_np = MagicMock()
mock_np.number = "number" # So np.number works

sys.modules["pandas"] = mock_pd
sys.modules["numpy"] = mock_np
sys.modules["streamlit"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["sklearn.feature_selection"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.decomposition"] = MagicMock()
sys.modules["sklearn.impute"] = MagicMock()
sys.modules["sklearn.neighbors"] = MagicMock()
sys.modules["sklearn.svm"] = MagicMock()
sys.modules["sklearn.ensemble"] = MagicMock()
sys.modules["sklearn.linear_model"] = MagicMock()
sys.modules["sklearn.naive_bayes"] = MagicMock()
sys.modules["sklearn.tree"] = MagicMock()
sys.modules["sklearn.neural_network"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["sklearn.mixture"] = MagicMock()
sys.modules["joblib"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["ydata_profiling"] = MagicMock()
sys.modules["xgboost"] = MagicMock()
sys.modules["imblearn"] = MagicMock()
sys.modules["imblearn.over_sampling"] = MagicMock()
sys.modules["pycaret"] = MagicMock()
sys.modules["pycaret.classification"] = MagicMock()
sys.modules["pycaret.regression"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Now import the app and the function
import app
from app import validate_dataframe

class TestValidateDataframe(unittest.TestCase):

    def test_df_none(self):
        """Test with None dataframe."""
        status, message = validate_dataframe(None)
        self.assertFalse(status)
        self.assertEqual(message, "No dataframe loaded")

    def test_df_empty(self):
        """Test with an empty dataframe."""
        mock_df = MagicMock()
        mock_df.empty = True
        status, message = validate_dataframe(mock_df)
        self.assertFalse(status)
        self.assertEqual(message, "Dataframe is empty")

    def test_df_less_than_10_rows(self):
        """Test with a dataframe having fewer than 10 rows."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.shape = (9, 3) # 9 rows, 3 columns
        status, message = validate_dataframe(mock_df)
        self.assertFalse(status)
        self.assertEqual(message, "Need at least 10 rows for training")

    def test_df_less_than_2_columns(self):
        """Test with a dataframe having fewer than 2 columns."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.shape = (10, 1) # 10 rows, 1 column
        status, message = validate_dataframe(mock_df)
        self.assertFalse(status)
        self.assertEqual(message, "Need at least 2 columns (features + target)")

    def test_df_valid(self):
        """Test with a valid dataframe."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.shape = (10, 2) # 10 rows, 2 columns
        status, message = validate_dataframe(mock_df)
        self.assertTrue(status)
        self.assertEqual(message, "Valid")

if __name__ == '__main__':
    unittest.main()
