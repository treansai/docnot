# Rust Medical Named Entity Recognition (NER) Server

This project implements a service for Named Entity Recognition (NER) system in Rust using the candle machine learning framework and 
blaze999/Medical-NER by Saketh Mattupalli[[https://huggingface.co/blaze999]]. 
## Features

- Load pre-trained NER models from HuggingFace
- Process text input to identify named entities
- Support for word embeddings and classification layers
- JSON output format with token positions and confidence scores

## Usage

### Basic Example
using build ready image
```yaml
# Pull the image
docker pull ghcr.io/treansai/docnot:latest

# Run the image
docker run -p 9494:9494 ghcr.io/treansai/docnot:latest
```

or self build

```yaml
# Build the image
docker build -t ner-server.

# Run the container
docker run -p 9494:9494 -v huggingface-cache:/root/.cache/huggingface ner-server

```

### Output Format

The model outputs predictions in JSON format:

```json
[
  {
    "token": "Sara",
    "label": "B-PER",
    "score": 0.9865,
    "start": 0,
    "end": 4
  },
  {
    "token": "London",
    "label": "B-LOC",
    "score": 0.9891,
    "start": 14,
    "end": 20
  }
]
```

## Model Architecture

The NER model uses a simple but effective architecture:

1. Word Embeddings Layer
2. Layer Normalization
3. Dropout for regularization
4. Linear classification layer

## Performance Considerations

- The model runs on CPU
- Input processing is done token by token
- Memory usage scales with input length
- Batch processing is not currently supported

## Contributing

Contributions are welcome! Here are some areas that could use improvement:

1. GPU support
2. Batch processing
3. Additional model architectures
4. Performance optimizations
5. More comprehensive testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace for providing pre-trained models
- Candle framework developers
- Saketh Mattupalli

## Support

For issues and questions:
1. Open an issue in the GitHub repository
2. Check the existing documentation
3. Review the error handling section

## Future Plans
- Add visualization tools
- ...