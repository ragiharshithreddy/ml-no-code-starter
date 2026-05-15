import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib

# Mock dependencies to avoid pollution and allow app import
mock_pd = MagicMock()
mock_np = MagicMock()
mock_np.number = "number"

# Create the dictionary of modules to mock
modules_to_mock = {
    "pandas": mock_pd,
    "numpy": mock_np,
    "streamlit": MagicMock(),
    "plotly": MagicMock(),
    "plotly.express": MagicMock(),
    "plotly.graph_objects": MagicMock(),
    "matplotlib": MagicMock(),
    "matplotlib.pyplot": MagicMock(),
    "seaborn": MagicMock(),
    "streamlit.components.v1": MagicMock(),
    "sklearn": MagicMock(),
    "sklearn.model_selection": MagicMock(),
    "sklearn.preprocessing": MagicMock(),
    "sklearn.feature_selection": MagicMock(),
    "sklearn.metrics": MagicMock(),
    "sklearn.decomposition": MagicMock(),
    "sklearn.impute": MagicMock(),
    "sklearn.neighbors": MagicMock(),
    "sklearn.svm": MagicMock(),
    "sklearn.ensemble": MagicMock(),
    "sklearn.linear_model": MagicMock(),
    "sklearn.naive_bayes": MagicMock(),
    "sklearn.tree": MagicMock(),
    "sklearn.neural_network": MagicMock(),
    "sklearn.cluster": MagicMock(),
    "sklearn.mixture": MagicMock(),
    "joblib": MagicMock(),
    "requests": MagicMock(),
    "ydata_profiling": MagicMock(),
    "xgboost": MagicMock(),
    "imblearn": MagicMock(),
    "imblearn.over_sampling": MagicMock(),
    "pycaret": MagicMock(),
    "pycaret.classification": MagicMock(),
    "pycaret.regression": MagicMock(),
    "transformers": MagicMock()
}

class TestCorrelationRecommendations(unittest.TestCase):

    def setUp(self):
        # We start the patcher for sys.modules
        self.patcher = patch.dict('sys.modules', modules_to_mock)
        self.patcher.start()

        # Now we can safely import app
        import app
        self.app = app

        # Reset st.error for each test
        self.app.st.error.reset_mock()

    def tearDown(self):
        # Stop the patcher to clean up sys.modules
        self.patcher.stop()

    def test_happy_path(self):
        """Test with a dataframe having clear correlations."""
        mock_df = MagicMock()
        mock_num_df = MagicMock()
        mock_corr = MagicMock()

        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 3)
        mock_num_df.corr.return_value = mock_corr

        mock_corr.columns = ['A', 'B', 'C']

        # Mocking corr.iloc[i, j]
        # matrix: A:B=0.9, A:C=-0.8, B:C=0.1
        matrix = {
            (0, 1): 0.9, (0, 2): -0.8,
            (1, 0): 0.9, (1, 2): 0.1,
            (2, 0): -0.8, (2, 1): 0.1,
            (0, 0): 1.0, (1, 1): 1.0, (2, 2): 1.0
        }
        mock_corr.iloc.__getitem__.side_effect = lambda x: matrix.get(x, 0.0)

        results = self.app.correlation_recommendations(mock_df, thresh=0.7)

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

        results = self.app.correlation_recommendations(mock_df)
        self.assertEqual(results, [])

    def test_single_numeric_column(self):
        """Test with only one numeric column."""
        mock_df = MagicMock()
        mock_num_df = MagicMock()
        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 1)

        results = self.app.correlation_recommendations(mock_df)
        self.assertEqual(results, [])

    def test_no_high_correlation(self):
        """Test when no features meet the threshold."""
        mock_df = MagicMock()
        mock_num_df = MagicMock()
        mock_corr = MagicMock()

        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 2)
        mock_num_df.corr.return_value = mock_corr
        mock_corr.columns = ['A', 'B']
        mock_corr.iloc.__getitem__.return_value = 0.5 # Below default 0.7

        results = self.app.correlation_recommendations(mock_df)
        self.assertEqual(results, [])

    def test_limit_15_results(self):
        """Test that it returns at most 15 recommendations."""
        mock_df = MagicMock()
        mock_num_df = MagicMock()
        mock_corr = MagicMock()

        mock_df.select_dtypes.return_value = mock_num_df
        mock_num_df.shape = (5, 20)
        mock_num_df.corr.return_value = mock_corr

        cols = [f'col_{i}' for i in range(20)]
        mock_corr.columns = cols
        mock_corr.iloc.__getitem__.return_value = 0.9

        results = self.app.correlation_recommendations(mock_df, thresh=0.5)
        # 20 columns -> 190 pairs. Should be limited to 15.
        self.assertEqual(len(results), 15)

    def test_exception_handling(self):
        """Test that exceptions are caught and reported."""
        mock_df = MagicMock()
        mock_df.select_dtypes.side_effect = Exception("Analysis Error")

        results = self.app.correlation_recommendations(mock_df)
        self.assertEqual(results, [])
        self.app.st.error.assert_called()
        self.assertIn("Analysis Error", self.app.st.error.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
