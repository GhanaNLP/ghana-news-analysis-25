# Ghana News Analysis 2025

A collaborative public information project to identify and visualize the **Top 50 Keywords** appearing in Ghanaian news throughout 2025.

## Project Overview

The goal is to scrape at least 5 major Ghanaian news outlets, process the text data, and perform keyword extraction to see what dominated the headlines and reports in 2025.

**Core Objectives:**

1. **Scrape** content published in 2025 from selected news sites.
2. **Clean and Tokenize** the data into sentences and words.
3. **Analyze** frequency to find the top 50 keywords.
4. **Visualize** the results for social media distribution.

------

## Tech Stack

- **Language:** Python 3.x
- **Scraping:** `newspaper3k` (for article parsing) or `BeautifulSoup`/`Scrapy`.
- **NLP & Tokenization:** `spaCy` (preferred) or `NLTK`.
- **Data Handling:** `pandas`.

------

## Contributors

Thanks to our contributors, we cover a wide selection of Ghanaian news sources.

| Contributor                                                  | News Site                                    |
| ------------------------------------------------------------ | -------------------------------------------- |
| [Kasuadana Sulemana Adams](https://www.linkedin.com/in/kasuadana1/) | [GhanaWeb](https://www.ghanaweb.com)         |
| [Obrempong Kwabena Osei-Wusu](https://www.linkedin.com/in/obrempong-kwabena-osei-wusu-7b0217257/) | [TV3 / 3News](https://3news.com)             |
| [Jonathan Ato Markin](https://www.linkedin.com/in/atomarkin/) | [Citi Newsroom](https://citinewsroom.com)    |
| [Josephus Bawah](https://www.linkedin.com/in/josephus-bawah/) | [Graphic Online](https://www.graphic.com.gh) |
| [Bernard Zephaniah](https://www.linkedin.com/in/bernard-zephaniah-addo-addo-6a728b220/) | [MyJoyOnline](https://www.myjoyonline.com)   |
| [Gerhardt Datsomor](https://www.linkedin.com/in/gerhardt-datsomor/) | [Ghana News Agency](https://gna.org.gh)      |

------

## Scraping Guidelines

When scraping your assigned site, please ensure:

- Only articles published between **January 1, 2025, and December 31, 2025**, are included.
- Data is saved in a CSV format with the following fields: `date`, `title`, `sentence`, and `url`.

### 3. Processing (Tokenization)

We are using **spaCy** or **nltk** for tokenisation.

------

## Communication

If you have any questions or want to contribute to the project, please send an email to natural.language.processing.gh@gmail.com.
