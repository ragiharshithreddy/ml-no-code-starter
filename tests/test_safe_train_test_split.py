import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock all dependencies to allow importing app.py without the actual packages
mock_pd = MagicMock()
mock_np = MagicMock()
mock_np.number = "number"

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
mock_train_test_split = MagicMock()
mock_model_selection = MagicMock()
mock_model_selection.train_test_split = mock_train_test_split
sys.modules["sklearn.model_selection"] = mock_model_selection
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

import app
from app import safe_train_test_split

class TestSafeTrainTestSplit(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        app.st.warning.reset_mock()
        app.st.error.reset_mock()
        app.train_test_split.reset_mock()
        app.np.unique.reset_mock()
        app.pd.Series.reset_mock()

    def test_classification_with_stratification(self):
        # We simulate a target array `y` where stratification is possible
        # i.e., at least 2 samples per class
        dummy_y = [0, 0, 1, 1]

        # We need np.unique(y) to return >1 class
        app.np.unique.return_value = [0, 1]

        # We need pd.Series(y).value_counts().min() to return >=2
        mock_series = MagicMock()
        app.pd.Series.return_value = mock_series
        mock_counts = MagicMock()
        mock_series.value_counts.return_value = mock_counts
        mock_counts.min.return_value = 2

        # train_test_split returns 4 values
        app.train_test_split.return_value = ("X_train", "X_test", "y_train", "y_test")

        res = safe_train_test_split("dummy_X", dummy_y, test_size=0.2, task="Classification")

        # Check return value
        self.assertEqual(res, ("X_train", "X_test", "y_train", "y_test"))
        # Check train_test_split called with stratify
        app.train_test_split.assert_called_once_with("dummy_X", dummy_y, test_size=0.2, random_state=42, stratify=dummy_y)
        # Check no warning was logged
        app.st.warning.assert_not_called()

    def test_classification_small_class(self):
        # We simulate an imbalanced target where one class has only 1 sample
        dummy_y = [0, 0, 1]

        app.np.unique.return_value = [0, 1]
        mock_series = MagicMock()
        app.pd.Series.return_value = mock_series
        mock_counts = MagicMock()
        mock_series.value_counts.return_value = mock_counts
        mock_counts.min.return_value = 1

        app.train_test_split.return_value = ("X_train", "X_test", "y_train", "y_test")

        res = safe_train_test_split("dummy_X", dummy_y, test_size=0.2, task="Classification")

        self.assertEqual(res, ("X_train", "X_test", "y_train", "y_test"))
        # Check train_test_split called without stratify
        app.train_test_split.assert_called_once_with("dummy_X", dummy_y, test_size=0.2, random_state=42)
        # Check warning was logged
        app.st.warning.assert_called_once()
        self.assertIn("Small class detected", app.st.warning.call_args[0][0])

    def test_regression_task(self):
        # Regression does not use stratification
        dummy_y = [0.1, 0.2, 0.3, 0.4]

        app.train_test_split.return_value = ("X_train", "X_test", "y_train", "y_test")

        res = safe_train_test_split("dummy_X", dummy_y, test_size=0.3, task="Regression")

        self.assertEqual(res, ("X_train", "X_test", "y_train", "y_test"))
        # Check train_test_split called without stratify
        app.train_test_split.assert_called_once_with("dummy_X", dummy_y, test_size=0.3, random_state=42)

    def test_classification_single_class(self):
        # Only one class present (len(np.unique(y)) == 1)
        dummy_y = [0, 0, 0]
        app.np.unique.return_value = [0]

        app.train_test_split.return_value = ("X_train", "X_test", "y_train", "y_test")

        res = safe_train_test_split("dummy_X", dummy_y, test_size=0.2, task="Classification")

        self.assertEqual(res, ("X_train", "X_test", "y_train", "y_test"))
        app.train_test_split.assert_called_once_with("dummy_X", dummy_y, test_size=0.2, random_state=42)

    def test_exception_handling(self):
        # Trigger an exception in np.unique
        app.np.unique.side_effect = Exception("Analysis Error")

        res = safe_train_test_split("dummy_X", "dummy_y", test_size=0.2, task="Classification")

        # Check fallback return
        self.assertEqual(res, (None, None, None, None))
        app.st.error.assert_called_once()
        self.assertIn("Split failed: Analysis Error", app.st.error.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
