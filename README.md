---

# Framewise SecureLine

Framewise SecureLine is a Python library for analyzing text and audio for security concerns. The library supports text-based detection and audio chunk predictions, offering flexibility for content security use cases.

## Features

- **Text Detection**: Analyze text for security concerns, such as banned topics, PII, or profanity.
- **Audio Chunk Analysis**: Process audio data in chunks for classification and security predictions.
- **Custom Filters**: Easily configure filters for denied topics, PII information, and word lists.
- **Extensibility**: Supports structured outputs for easy integration with your workflows.
- **Robust Error Handling**: Handles API errors, validation errors, and request timeouts gracefully.

---

## Installation

Framewise SecureLine is available for installation using **Poetry** or **pip**.

### Install using Poetry
```bash
poetry add framewise-secureline
```

### Install using pip
```bash
pip install framewise-secureline
```

---

## Getting Started

### Initialization

To use Framewise SecureLine, you need an API key. Initialize the `SecureLine` client as shown below:

```python
from framewise_secureline import SecureLine

# Initialize the client
client = SecureLine(
    api_key="your-api-key",
    denied_topics="medical diagnoses, competitors",
    ppi_information="SSN, Medical Records",
    word_filters="profanity",
)
```

---

### Text Detection

Analyze a piece of text for security concerns.

```python
result = client.detect("Sample text for analysis.")

# Access raw response
print(result)
```

---

### Audio Chunk Analysis

Analyze an audio chunk for predictions. The library accepts a single chunk of type `AudioSegment`.

```python
from pydub import AudioSegment

# Load an audio file and extract a chunk
audio = AudioSegment.from_file("/path/to/audio.mpga").set_frame_rate(16000).set_channels(1)
chunk = audio[:100]  # Extract first 100ms

# Analyze the chunk
result = client.analyze_audio_chunk(chunk)

# Access predictions
for prediction in result.predictions:
    print(f"Label: {prediction.label}, Score: {prediction.score}")
print(f"Latency: {result.latency}, Device: {result.device}")
```

---

## Advanced Usage

### Custom Filters

You can update the filters dynamically using the `update_filters` method:

```python
client.update_filters(
    denied_topics="new denied topics",
    ppi_information="new PII types",
    word_filters="new word list",
)
```

### Handling Multiple Audio Chunks

While SecureLine processes one audio chunk at a time, you can handle multiple chunks by iterating over the audio file:

```python
chunks = [audio[i:i+100] for i in range(0, len(audio), 100)]  # Split into 100ms chunks

for i, chunk in enumerate(chunks):
    result = client.analyze_audio_chunk(chunk)
    print(f"Chunk {i}:")
    for prediction in result.predictions:
        print(f"  Label: {prediction.label}, Score: {prediction.score}")
```

---

## Error Handling

Framewise SecureLine provides robust error handling for common scenarios:

- **ValidationError**: Raised for invalid inputs (e.g., empty text, incorrect audio format).
- **APIError**: Raised for server-side errors during API calls.
- **TimeoutError**: Raised when an API call exceeds the timeout.

### Example

```python
from framewise_secureline.exceptions import ValidationError, APIError, TimeoutError

try:
    result = client.detect("text to analyze")
except ValidationError as e:
    print(f"Validation failed: {e}")
except APIError as e:
    print(f"API error: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
```

---

## Testing

The library includes a comprehensive suite of tests.

### Run Tests

```bash
poetry run pytest
```

### Coverage Report

```bash
poetry run pytest --cov=framewise_secureline
```

---

## Support

For any issues or feature requests, please contact:  
**Email**: sigireddybalasai@gmail.com

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---