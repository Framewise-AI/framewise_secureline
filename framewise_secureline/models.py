import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Callable, Tuple, Any, List
from .exceptions import ValidationError
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """Enum to define risk categories."""
    BENIGN = "Benign"
    PROMPT_ATTACKS = "Prompt Attacks"
    DENIED_TOPICS = "Denied Topics"
    PPI = "PPI Information"
    WORD_FILTERS = "Word Filters"


@dataclass
class DetectionResult:
    """Container for detection results."""
    raw_response: Dict[str, Dict[str, float]]
    elapsed_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    _probabilities: Dict[str, float] = field(init=False)

    def __post_init__(self):
        logger.info("Initializing DetectionResult instance.")
        self._probabilities = self.raw_response.get("probabilities", {})
        if not isinstance(self._probabilities, dict):
            logger.error("Invalid probabilities format in raw_response.")
            raise ValidationError("Probabilities must be a dictionary.")
        logger.info("Initialization complete. Probabilities loaded.")

    def get_score(self, category: RiskCategory) -> float:
        """Get score for a specific category."""
        logger.info(f"Fetching score for category: {category.name}.")
        if not isinstance(category, RiskCategory):
            logger.error(f"Invalid category provided: {category}.")
            raise ValidationError("Invalid category.")
        score = self._probabilities.get(category.value, 0.0)
        logger.info(f"Score for {category.name}: {score}.")
        return score

    def get_highest_risk(self) -> Tuple[str, float]:
        """Get the highest risk category."""
        if not self._probabilities:
            logger.warning("No probabilities available for highest risk calculation.")
            return "None", 0.0
        highest_risk = max(self._probabilities.items(), key=lambda x: x[1], default=("None", 0.0))
        logger.info(f"Highest risk category: {highest_risk[0]} with score: {highest_risk[1]}.")
        return highest_risk

    def __getattr__(self, name: str) -> Callable[[], float]:
        """
        Dynamically handle get_{category} calls as functions.

        Args:
            name (str): The name of the method being accessed.

        Returns:
            Callable[[], float]: A function that returns the score for the requested category.

        Raises:
            ValidationError: If the requested method does not match the expected pattern.
        """
        if name.startswith("get_"):
            category_name = name[4:].upper()
            try:
                category = RiskCategory[category_name]
                logger.info(f"Creating dynamic method for category: {category_name}.")
                return lambda: self.get_score(category)
            except KeyError:
                logger.error(f"Invalid risk category requested: {category_name}.")
                raise ValidationError(f"Invalid risk category: {category_name}")
        logger.error(f"Invalid attribute or method requested: {name}.")
        raise AttributeError(f"'DetectionResult' object has no attribute '{name}'")


@dataclass
class Prediction:
    """Single prediction result with score and label
    
    Attributes:
        label (str): Prediction label (e.g., 'fake' or 'real')
        score (float): Confidence score for this label
    """
    label: str
    score: float

@dataclass
class AudioResult:
    """Results from processing an audio chunk
    
    Attributes:
        raw_response (Dict[str, Any]): Raw API response containing predictions and metadata
        elapsed_time (float): Processing time in milliseconds
        processed_at (datetime): Timestamp when processing completed
        
    Example:
        >>> response = {
        ...     "predictions": [
        ...         {"score": 0.9999, "label": "fake"},
        ...         {"score": 0.0001, "label": "real"}
        ...     ],
        ...     "latency": 0.011,
        ...     "device": "GPU"
        ... }
        >>> result = AudioResult(
        ...     raw_response=response,
        ...     elapsed_time=150.5,
        ...     processed_at=datetime.now()
        ... )
        >>> print(result.predictions[0].label)  # Output: "fake"
        >>> print(result.predictions[0].score)  # Output: 0.9999
    """
    raw_response: Dict[str, Any]
    elapsed_time: float
    processed_at: datetime

    @property
    def predictions(self) -> List[Prediction]:
        """Get list of predictions sorted by score in descending order"""
        preds = self.raw_response.get('predictions', [])
        return [
            Prediction(
                label=p['label'],
                score=p['score']
            ) for p in preds
        ]
    
    @property
    def top_prediction(self) -> Prediction:
        """Get the prediction with highest confidence score"""
        preds = self.predictions
        return preds[0] if preds else Prediction(label='unknown', score=0.0)
    
    @property
    def latency(self) -> float:
        """Get server-side processing latency in seconds"""
        return self.raw_response.get('latency', 0.0)
    
    @property
    def device(self) -> str:
        """Get processing device type (e.g., 'GPU')"""
        return self.raw_response.get('device', '')