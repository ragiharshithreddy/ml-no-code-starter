import unittest
from unittest.mock import MagicMock, patch
import sys
import smtplib
import importlib

# Mock dependencies to avoid pollution and allow app import
mock_pd = MagicMock()
mock_np = MagicMock()
mock_np.number = "number"

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

class TestSendResultsEmail(unittest.TestCase):

    def setUp(self):
        # We start the patcher for sys.modules
        self.patcher = patch.dict('sys.modules', modules_to_mock)
        self.patcher.start()

        # Now we can safely import app
        import app
        # Force reload app so it uses the mocks defined above when it initializes.
        importlib.reload(app)
        self.app = app

        # Reset mocks before each test
        self.app.st.error.reset_mock()
        self.app.st.warning.reset_mock()
        # Save original config
        self.original_enable = self.app.ENABLE_EMAIL

    def tearDown(self):
        # Restore original config
        self.app.ENABLE_EMAIL = self.original_enable
        # Stop the patcher to clean up sys.modules
        self.patcher.stop()

    def test_email_disabled(self):
        self.app.ENABLE_EMAIL = False
        result = self.app.send_results_email("test@test.com", "Subject", {})
        self.assertFalse(result)
        self.app.st.warning.assert_called_once_with("📧 Email feature is disabled. Configure credentials to enable.")

    def test_invalid_email_format(self):
        self.app.ENABLE_EMAIL = True

        # Test empty
        result = self.app.send_results_email("", "Subject", {})
        self.assertFalse(result)
        self.app.st.error.assert_called_with("❌ Invalid email address format.")

        # Test missing @
        result = self.app.send_results_email("test.com", "Subject", {})
        self.assertFalse(result)

        # Test missing .
        result = self.app.send_results_email("test@com", "Subject", {})
        self.assertFalse(result)

    @patch("app.smtplib.SMTP_SSL")
    @patch("app.ssl.create_default_context")
    def test_successful_email(self, mock_ssl_context, mock_smtp_ssl):
        self.app.ENABLE_EMAIL = True

        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server

        result = self.app.send_results_email("test@test.com", "Subject", {"model": "RF", "task": "Classification", "accuracy": 0.95})

        self.assertTrue(result)
        mock_server.login.assert_called_once_with(self.app.OWNER_GMAIL, self.app.OWNER_APP_PASSWORD)
        mock_server.sendmail.assert_called_once()
        args, kwargs = mock_server.sendmail.call_args
        self.assertEqual(args[0], self.app.OWNER_GMAIL)
        self.assertEqual(args[1], "test@test.com")
        self.assertIn("Subject", args[2])

    @patch("app.smtplib.SMTP_SSL")
    def test_smtp_auth_error(self, mock_smtp_ssl):
        self.app.ENABLE_EMAIL = True

        # Make the context manager __enter__ raise the error
        mock_smtp_ssl.return_value.__enter__.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")

        result = self.app.send_results_email("test@test.com", "Subject", {})
        self.assertFalse(result)
        self.app.st.error.assert_called_once_with("❌ Email authentication failed. Check credentials.")

    @patch("app.smtplib.SMTP_SSL")
    def test_smtp_exception(self, mock_smtp_ssl):
        self.app.ENABLE_EMAIL = True

        mock_smtp_ssl.return_value.__enter__.side_effect = smtplib.SMTPException("Some SMTP error")

        result = self.app.send_results_email("test@test.com", "Subject", {})
        self.assertFalse(result)
        self.app.st.error.assert_called_once_with("❌ Email sending failed: Some SMTP error")

    @patch("app.smtplib.SMTP_SSL")
    def test_general_exception(self, mock_smtp_ssl):
        self.app.ENABLE_EMAIL = True

        mock_smtp_ssl.return_value.__enter__.side_effect = Exception("General error")

        result = self.app.send_results_email("test@test.com", "Subject", {})
        self.assertFalse(result)
        self.app.st.error.assert_called_once_with("❌ Unexpected error: General error")

if __name__ == '__main__':
    unittest.main()
