import os
import pytest
from unittest.mock import patch, Mock
from datetime import datetime
from dotenv import load_dotenv
from framewise_secureline import SecureLine
from framewise_secureline.exceptions import ValidationError, APIError, TimeoutError
from pydub import AudioSegment
from framewise_secureline.models import AudioResult
load_dotenv()

@pytest.fixture
def api_key():
    """Get API key from environment variable or use a test key"""
    key = os.getenv("SECURELINE_API_KEY")
    if not key:
        pytest.skip("SECURELINE_API_KEY environment variable not set")
    return key

@pytest.fixture
def client(api_key):
    """Create a SecureLine client instance"""
    return SecureLine(
        api_key=api_key,
        denied_topics="test topics",
        ppi_information="test ppi",
        word_filters="test filters"
    )

@pytest.fixture
def mock_response():
    """Create a mock successful API response"""
    return {
        "probabilities": {
            "Benign": 0.9,
            "Prompt Attacks": 0.1,
            "Denied Topics": 0.0,
            "PPI Information": 0.0,
            "Word Filters": 0.0
        }
    }

def test_client_initialization(api_key):
    """Test client initialization with API key from environment"""
    client = SecureLine(api_key=api_key)
    assert client.api_key == api_key
    assert client.denied_topics == ""
    assert client.ppi_information == ""
    assert client.word_filters == ""
    assert client.timeout == 10

def test_client_initialization_with_filters(api_key):
    """Test client initialization with custom filters"""
    client = SecureLine(
        api_key=api_key,
        denied_topics="medical",
        ppi_information="ssn",
        word_filters="profanity"
    )
    assert client.denied_topics == "medical"
    assert client.ppi_information == "ssn"
    assert client.word_filters == "profanity"

def test_update_filters(client):
    """Test updating client filters"""
    client.update_filters(
        denied_topics="new topics",
        ppi_information="new ppi",
        word_filters="new filters"
    )
    assert client.denied_topics == "new topics"
    assert client.ppi_information == "new ppi"
    assert client.word_filters == "new filters"

def test_detect_validation_error(client):
    """Test validation error when text is empty"""
    with pytest.raises(ValidationError, match="Text must be a non-empty string"):
        client.detect("")

def test_detect_api_error(client):
    """Test API error handling"""
    with patch('requests.request') as mock_request:
        mock_request.side_effect = Exception("API error")
        
        with pytest.raises(APIError, match="Request failed with no specific error: API error"):
            client.detect("test text")


def test_detect_timeout_error(client):
    """Test timeout error handling"""
    with patch('requests.request') as mock_request:
        mock_request.side_effect = TimeoutError("Timeout")
        
        with pytest.raises(TimeoutError):
            client.detect("test text")

@pytest.mark.vcr()
def test_detect_success(client, mock_response):
    """Test successful detection request"""
    with patch('requests.request') as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {
            "probabilities": {
                "Benign": 0.9,
                "Prompt Attacks": 0.1,
                "Denied Topics": 0.0,
                "PPI Information": 0.0,
                "Word Filters": 0.0
            }
        }
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        result = client.detect("test text")
        
        assert result.raw_response["probabilities"]["Benign"] == 0.9
        assert result.raw_response["probabilities"]["Prompt Attacks"] == 0.1


def test_detect_with_retries(client):
    """Test retry mechanism"""
    with patch('requests.request') as mock_request:
        # First two calls fail, third succeeds
        mock_request.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            Mock(
                status_code=200,
                json=lambda: {"probabilities": {"Benign": 1.0}}
            )
        ]
        
        result = client.detect("test text", retry_count=3, retry_delay=0)
        assert result.raw_response["probabilities"]["Benign"] == 1.0


@pytest.mark.integration
def test_integration_detect(client):
    """Test actual API integration (requires API key)"""
    result = client.detect("This is a test message")
    assert isinstance(result._probabilities, dict)
    assert "Benign" in result._probabilities
    assert isinstance(result.elapsed_time, float)



@pytest.mark.integration
def test_real_request_detection(client):
    """
    Integration test to detect a message and validate probabilities.
    """
    try:
        # Example message for detection
        message = "This is a harmless test message."

        # Send a request to the SecureLine API
        result = client.detect(message)

        # Validate the structure of the response
        assert isinstance(result.raw_response, dict), "Response should be a dictionary."
        assert "probabilities" in result.raw_response, "'probabilities' should be in the response."
        probabilities = result.raw_response["probabilities"]
        assert isinstance(probabilities, dict), "'probabilities' should be a dictionary."
        
        # Check for expected categories
        expected_categories = ["Benign", "Prompt Attacks", "Denied Topics", "PPI Information", "Word Filters"]
        for category in expected_categories:
            assert category in probabilities, f"'{category}' should be in probabilities."
            assert isinstance(probabilities[category], float), f"Probability for '{category}' should be a float."

        # Example: Validate that the message is classified as benign
        benign_score = probabilities["Benign"]
        assert benign_score > 0.5, "Message should be classified as benign."

        # Log the result
        print("Detection successful. Probabilities:", probabilities)

    except (ValidationError, APIError, TimeoutError) as e:
        pytest.fail(f"Real request test failed with exception: {e}")

from datetime import datetime
import pytest
from framewise_secureline.client import AudioResult


@pytest.fixture
def sample_response():
    """Sample response for testing."""
    return {
        "predictions": [
            {"label": "fake", "score": 0.9999746084213257},
            {"label": "real", "score": 0.00002542001129768323},
        ],
        "latency": 0.01098942756652832,
        "device": "GPU",
    }


class TestAudioResult:
    def test_predictions_parsing(self, sample_response):
        """Test parsing of predictions from response."""
        result = AudioResult(
            raw_response=sample_response,
            elapsed_time=150.5,
            processed_at=datetime.now(),
        )

        assert len(result.predictions) == 2

    def test_top_prediction(self, sample_response):
        """Test top prediction accessor."""
        result = AudioResult(
            raw_response=sample_response,
            elapsed_time=150.5,
            processed_at=datetime.now(),
        )

        top = result.top_prediction
        assert top.label == "fake"
        assert top.score == pytest.approx(0.9999746084213257)

    def test_metadata(self, sample_response):
        """Test metadata accessors."""
        result = AudioResult(
            raw_response=sample_response,
            elapsed_time=150.5,
            processed_at=datetime.now(),
        )

        assert result.latency == pytest.approx(0.01098942756652832)
        assert result.device == "GPU"

    @pytest.mark.parametrize(
        "response,expected_label,expected_score",
        [
            (
                {
                    "predictions": [
                        {"score": 0.9, "label": "fake"},
                        {"score": 0.1, "label": "real"},
                    ]
                },
                "fake",
                0.9,
            ),
            (
                {
                    "predictions": [
                        {"score": 0.8, "label": "real"},
                        {"score": 0.2, "label": "fake"},
                    ]
                },
                "real",
                0.8,
            ),
        ],
    )
    def test_different_predictions(self, response, expected_label, expected_score):
        """Test handling of different prediction patterns."""
        result = AudioResult(
            raw_response=response,
            elapsed_time=150.5,
            processed_at=datetime.now(),
        )

        assert result.top_prediction.label == expected_label
        assert result.top_prediction.score == pytest.approx(expected_score)
