import unittest
from unittest.mock import MagicMock, patch
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
from app import correlation_recommendations

class TestCorrelationRecommendations(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test
        app.st.error.reset_mock()

    def test_happy_path(self):
        """Test with a dataframe having clear correlations."""
        import sys
        if 'pandas' in sys.modules and isinstance(sys.modules['pandas'], MagicMock):
            del sys.modules['pandas']
        if 'numpy' in sys.modules and isinstance(sys.modules['numpy'], MagicMock):
            del sys.modules['numpy']

        import pandas as pd
        import numpy as np

        mock_df = MagicMock()
        mock_num_df = MagicMock()

        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 3)

        # We need a real pandas DataFrame for `corr.where` and `stack` to work correctly.
        mock_corr = pd.DataFrame(
            np.array([[1.0, 0.9, -0.8], [0.9, 1.0, 0.1], [-0.8, 0.1, 1.0]]),
            columns=['A', 'B', 'C'], index=['A', 'B', 'C']
        )
        mock_num_df.corr.return_value = mock_corr

        app.pd = pd
        app.np = np

        results = correlation_recommendations(mock_df, thresh=0.7)

        # Expected: (A, B, 0.9), (A, C, -0.8) sorted by abs value
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], ('A', 'B', 0.9))
        self.assertEqual(results[1], ('A', 'C', -0.8))

    def test_no_numeric_columns(self):
        """Test with no numeric columns."""
        mock_df = MagicMock()
        mock_num_df = MagicMock()
        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 0)

        results = correlation_recommendations(mock_df)
        self.assertEqual(results, [])

    def test_single_numeric_column(self):
        """Test with only one numeric column."""
        mock_df = MagicMock()
        mock_num_df = MagicMock()
        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 1)

        results = correlation_recommendations(mock_df)
        self.assertEqual(results, [])

    def test_no_high_correlation(self):
        """Test when no features meet the threshold."""
        import sys
        if 'pandas' in sys.modules and isinstance(sys.modules['pandas'], MagicMock):
            del sys.modules['pandas']
        if 'numpy' in sys.modules and isinstance(sys.modules['numpy'], MagicMock):
            del sys.modules['numpy']

        import pandas as pd
        import numpy as np

        mock_df = MagicMock()
        mock_num_df = MagicMock()

        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 2)

        mock_corr = pd.DataFrame(
            np.array([[1.0, 0.5], [0.5, 1.0]]),
            columns=['A', 'B'], index=['A', 'B']
        )
        mock_num_df.corr.return_value = mock_corr

        app.pd = pd
        app.np = np

        results = correlation_recommendations(mock_df)
        self.assertEqual(results, [])

    def test_limit_15_results(self):
        """Test that it returns at most 15 recommendations."""
        import sys
        if 'pandas' in sys.modules and isinstance(sys.modules['pandas'], MagicMock):
            del sys.modules['pandas']
        if 'numpy' in sys.modules and isinstance(sys.modules['numpy'], MagicMock):
            del sys.modules['numpy']

        import pandas as pd
        import numpy as np

        mock_df = MagicMock()
        mock_num_df = MagicMock()

        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 20)

        cols = [f'col_{i}' for i in range(20)]
        mock_corr = pd.DataFrame(np.full((20, 20), 0.9), columns=cols, index=cols)
        np.fill_diagonal(mock_corr.values, 1.0)
        mock_num_df.corr.return_value = mock_corr

        app.pd = pd
        app.np = np

        results = correlation_recommendations(mock_df, thresh=0.5)
        # 20 columns -> 190 pairs. Should be limited to 15.
        self.assertEqual(len(results), 15)

    def test_exception_handling(self):
        """Test that exceptions are caught and reported."""
        mock_df = MagicMock()
        mock_df.select_dtypes.side_effect = Exception("Analysis Error")

        results = correlation_recommendations(mock_df)
        self.assertEqual(results, [])
        app.st.error.assert_called()
        self.assertIn("Analysis Error", app.st.error.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
