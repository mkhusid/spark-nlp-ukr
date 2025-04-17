# Spark NLP for Ukrainian Text Processing

This project demonstrates the use of Spark NLP and PySpark for processing and analyzing Ukrainian text data. It includes steps for text normalization, tokenization, lemmatization, feature extraction, and advanced text analysis such as cosine similarity, duplicate detection, and topic modeling using LDA.

## Features

- **Text Preprocessing**: Cleaning, tokenization, and lemmatization of Ukrainian text.
- **Feature Extraction**: Bag of Words, TF-IDF, and Word2Vec embeddings.
- **Similarity Analysis**: Cosine similarity between text fields.
- **Duplicate Detection**: Identifying duplicate records based on similarity thresholds.
- **Topic Modeling**: LDA-based topic extraction.
- **Integration with Spark NLP**: Leveraging pretrained models for embeddings and stopword removal.

## Setup

### Prerequisites

- Python 3.8+
- Apache Spark 3.x
- PySpark
- Spark NLP
- Spacy with the `uk_core_news_sm` model
- NumPy and Pandas

### Installation

1. Install Python dependencies:
   ```bash
   pip install pyspark sparknlp spacy numpy pandas
   ```

2. Download and install the Spacy model for Ukrainian:
   ```bash
   python -m spacy download uk_core_news_sm
   ```

3. Ensure Spark NLP is included in your Spark session:
   ```bash
   --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.2
   ```

## Usage

### Data Preparation

Place your input CSV file containing Ukrainian text data in the `./data/` directory. The file should have at least two columns: `Title` and `Body`.

### Running the Notebook

1. Open the Jupyter Notebook `spark-nlp-ukr.ipynb`.
2. Follow the cells step-by-step to:
   - Preprocess the text data.
   - Extract features like Bag of Words and TF-IDF.
   - Perform similarity analysis and duplicate detection.
   - Generate embeddings using Spark NLP.
   - Build LDA models for topic extraction.

### Key Classes and Functions

- **`UkrainianProcessor`**: A class for preprocessing Ukrainian text, including cleaning, tokenization, lemmatization, and feature extraction.
- **`make_embedding`**: A function to generate Word2Vec embeddings using Spark NLP.
- **`build_lda`**: A function to create LDA models for topic modeling.

### Example Workflow

```python
from pyspark.sql import SparkSession
from spark_nlp_ukr import UkrainianProcessor, make_embedding, build_lda

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Ukrainian NLP") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.2") \
    .getOrCreate()

# Load data
data = spark.read.option("header", "true").csv("./data/ukr_text.csv")

# Preprocess text
processor = UkrainianProcessor(data)
preprocessed_body = processor.preprocessing("body")

# Generate embeddings
embedded_body = make_embedding(preprocessed_body, "body")

# Build LDA model
build_lda(5, 10, processor.words, preprocessed_body, "body_features")
```

## Results

- **Cosine Similarity**: Measure the similarity between `Title` and `Body` fields.
- **Duplicate Detection**: Identify duplicate records based on similarity thresholds.
- **Topic Modeling**: Extract topics from text using LDA.

## Directory Structure

```
spark-nlp-ukr/
├── data/
│   └── ukr_text.csv         # Input data
├── spark-nlp-ukr.ipynb      # Jupyter Notebook
├── README.md                # Project documentation
```

## References

- [Spark NLP Documentation](https://nlp.johnsnowlabs.com/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Spacy Documentation](https://spacy.io/)

## License

This project is licensed under the MIT License.
