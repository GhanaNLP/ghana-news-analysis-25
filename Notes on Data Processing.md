# Notes on Data Processing

## Overview

This document outlines the complete data processing pipeline used to extract and analyze keywords from Ghana News content for 2025. The methodology employed a multi-stage filtering and classification approach to identify concrete, development-related keywords from a large corpus of news articles.

------

## Data Sources

### News Sites Coverage

Five major Ghanaian news outlets were scraped for content published in 2025:

| News Site         | Total Sentences | Total Articles | Unique Noun Phrases | Total Phrase Occurrences | Avg Phrases/Article |
| ----------------- | --------------- | -------------- | ------------------- | ------------------------ | ------------------- |
| Citi Newsroom     | 246,749         | 15,422         | 327,220             | 1,418,186                | 21.22               |
| Ghana News Agency | 150,869         | 10,752         | 260,448             | 1,065,282                | 24.22               |
| GhanaWeb          | 129,335         | 7,908          | 142,791             | 701,202                  | 18.06               |
| Graphic Online    | 127,741         | 5,321          | 222,474             | 724,646                  | 41.81               |
| MyJoyOnline       | 304,808         | 21,615         | 440,307             | 2,040,161                | 20.37               |
| **Total**         | **959,502**     | **61,018**     | **1,393,240**       | **5,949,477**            | **21.71**           |

------

## Processing Pipeline

### Content Extraction

- Extracted news articles from all 5 news websites covering the 2025 calendar year
- Articles saved with metadata including date, title, and URL

### Sentence Tokenization

- Tokenized all articles into individual sentences using spaCy
- Created sentence-level datasets for each news source
- Output files: `*_tokenized.csv` files in `/sentence-datasets/`

### Noun Phrase Extraction

- Used spaCy NLP library to extract nouns and noun phrases from each sentence
- Counted frequency of each noun phrase across all articles
- Output files: `*_noun_phrases.csv` files in `/noun-phrases/` directory

**Filtering Criteria:**

- Noun phrases must appear in at least 20 articles across the corpus
- This filter helped to focus on topics that appear at a reasonable frequency across articles.

### Concept Classification

- Used Llama4-Maverick model to classify noun phrases as:

  - **Concepts:** Abstract or concrete ideas worthy of analysis
  - **Non-concepts:** Generic terms, stop words, or irrelevant phrases
  
- Filtered dataset to retain only classified concepts

### Keyword Classification

- Used Kimi-K2-Instruct-0905 model to further classify concepts into:

  - **Keywords:** Concrete, development-related activities (e.g., infrastructure projects, policy initiatives, economic activities)
  - **Non-keywords:** Abstract concepts, non-developmental activities, or generic terms
  
- Retained only keywords for final analysis

### Manual Review and Final Cleaning

Manual review was conducted to remove remaining noise and determine appropriate cut-off thresholds.

**Specific Cleaning Actions:**

1. Noise Removal:

    The following generic terms were removed as they added no analytical value:

   - "Project"
   - "Programme"
   - "Job"
   - "Design"

2. Term Merging:

    Related terms were consolidated to avoid duplication:

   - "Galamsey" and "illegal mining" were merged into a single keyword

3. **Threshold Determination:** Final frequency cut-off was established based on distribution analysis and relevance

------

## Technical Implementation

### Tools Used

- **spaCy:** NLP processing and noun phrase extraction
- **Llama4-Maverick:** Concept classification
- **Kimi-K2-Instruct-0905:** Keyword classification
- **pandas:** Data manipulation and analysis

------

## Key Statistics

- **Total sentences processed:** 959,502
- **Total articles analyzed:** 61,018
- **Initial unique noun phrases:** 1,393,240
- **Total phrase occurrences:** 5,949,477
- **Average phrases per article:** 21.71

------

