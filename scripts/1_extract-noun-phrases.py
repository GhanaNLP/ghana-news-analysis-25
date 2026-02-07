#!/usr/bin/env python3
"""
Extract noun phrases from multiple CSVs with per-article deduplication.
Counts each noun phrase only once per article (same title + date).
Outputs detailed statistics per CSV and overall summary.
"""
import os
import glob
import pandas as pd
from tqdm import tqdm
import spacy
from collections import defaultdict

# ------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------
INPUT_DIR = "/home/owusus/Documents/GitHub/ghana-news-analysis/sentence-datasets/"  # Directory with CSVs
INPUT_PATTERN = "*.csv"  # Pattern to match CSV files
TEXT_COLUMN = "sentence"
TITLE_COLUMN = "title"
DATE_COLUMN = "date"
URL_COLUMN = "url"
OUTPUT_DIR = "/home/owusus/Documents/GitHub/ghana-news-analysis/noun-phrases/"
BATCH_SIZE = 5000
SAVE_EVERY = 25000

# ------------------------------------------------------------------
# Load SpaCy model
print("🔤 Loading SpaCy model ('en_core_web_sm')...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
stop_words = nlp.Defaults.stop_words
print(f"📋 Loaded {len(stop_words)} stop words")

# ------------------------------------------------------------------
# Helper function
# ------------------------------------------------------------------
def strip_leading_stopwords(phrase, stop_words):
    """Remove stop words from the beginning of a phrase."""
    words = phrase.split()
    while words and words[0].lower() in stop_words:
        words.pop(0)
    return " ".join(words)

# ------------------------------------------------------------------
# Find all CSV files
# ------------------------------------------------------------------
# Use os.path.join to look in the correct directory
search_path = os.path.join(INPUT_DIR, INPUT_PATTERN)
csv_files = sorted(glob.glob(search_path))

if not csv_files:
    # Try current directory as fallback
    csv_files = sorted(glob.glob(INPUT_PATTERN))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{INPUT_DIR}' or current directory matching pattern: {INPUT_PATTERN}")

print(f"📂 Found {len(csv_files)} CSV files to process:")
for f in csv_files:
    print(f"   - {os.path.basename(f)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Process each CSV
# ------------------------------------------------------------------
all_results = []
overall_stats = []

for csv_file in csv_files:
    print(f"\n{'='*60}")
    print(f"📄 Processing: {os.path.basename(csv_file)}")
    print(f"{'='*60}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    csv_name = os.path.basename(csv_file).replace('.csv', '')
    
    # Validate columns
    missing_cols = [col for col in [TEXT_COLUMN, TITLE_COLUMN, DATE_COLUMN] if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Skipping {csv_file}: missing columns {missing_cols}")
        continue
    
    # Drop NaN sentences
    df = df.dropna(subset=[TEXT_COLUMN])
    total_sentences = len(df)
    
    print(f"📊 Total sentences: {total_sentences}")
    
    # Group by article (title + date) for per-article deduplication
    article_groups = df.groupby([TITLE_COLUMN, DATE_COLUMN])
    total_articles = len(article_groups)
    print(f"📰 Unique articles: {total_articles}")
    
    # Track noun phrases per article (for deduplication)
    article_noun_phrases = defaultdict(lambda: defaultdict(tuple))
    
    # Process sentences
    for (title, date), group in tqdm(article_groups, desc=f"Processing articles in {csv_name}", total=total_articles):
        sentences = group[TEXT_COLUMN].astype(str).str.strip().tolist()
        article_key = (title, date)
        
        # Process all sentences for this article
        docs = list(nlp.pipe(sentences, disable=["ner"]))
        
        for doc in docs:
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip()
                if not phrase:
                    continue
                    
                # Strip leading stop words
                phrase = strip_leading_stopwords(phrase, stop_words)
                if not phrase:
                    continue
                
                # Use lowercase as key for deduplication
                phrase_key = phrase.lower()
                
                # Store only once per article (increment count if already seen)
                if phrase_key in article_noun_phrases[article_key]:
                    orig, count = article_noun_phrases[article_key][phrase_key]
                    # Prefer version with uppercase
                    if phrase != phrase_key and orig == phrase_key:
                        orig = phrase
                    article_noun_phrases[article_key][phrase_key] = (orig, count + 1)
                else:
                    article_noun_phrases[article_key][phrase_key] = (phrase, 1)
    
    # Aggregate results for this CSV (sum counts across all articles)
    csv_noun_dict = {}
    for article_phrases in article_noun_phrases.values():
        for phrase_key, (orig_phrase, count) in article_phrases.items():
            if phrase_key in csv_noun_dict:
                old_orig, old_count = csv_noun_dict[phrase_key]
                # Prefer version with uppercase
                if orig_phrase != phrase_key and old_orig == phrase_key:
                    old_orig = orig_phrase
                csv_noun_dict[phrase_key] = (old_orig, old_count + count)
            else:
                csv_noun_dict[phrase_key] = (orig_phrase, count)
    
    # Create DataFrame for this CSV
    csv_df = pd.DataFrame([
        {"phrase": orig, "pos": "NOUN_PHRASE", "count": c, "csv_source": csv_name}
        for (orig, c) in csv_noun_dict.values()
    ]).sort_values("count", ascending=False).reset_index(drop=True)
    
    # Save individual CSV results
    output_file = os.path.join(OUTPUT_DIR, f"{csv_name}_noun_phrases.csv")
    csv_df.to_csv(output_file, index=False)
    
    # Calculate stats
    unique_phrases = len(csv_df)
    total_phrase_occurrences = csv_df["count"].sum()
    avg_phrases_per_article = unique_phrases / total_articles if total_articles > 0 else 0
    
    stats = {
        "csv_file": csv_file,
        "csv_name": csv_name,
        "total_sentences": total_sentences,
        "total_articles": total_articles,
        "unique_noun_phrases": unique_phrases,
        "total_phrase_occurrences": int(total_phrase_occurrences),
        "avg_phrases_per_article": round(avg_phrases_per_article, 2),
        "output_file": output_file
    }
    overall_stats.append(stats)
    all_results.append(csv_df)
    
    print(f"\n📈 Stats for {csv_name}:")
    print(f"   - Sentences processed: {total_sentences:,}")
    print(f"   - Articles (unique title+date): {total_articles:,}")
    print(f"   - Unique noun phrases: {unique_phrases:,}")
    print(f"   - Total phrase occurrences: {int(total_phrase_occurrences):,}")
    print(f"   - Avg unique phrases per article: {avg_phrases_per_article:.2f}")
    print(f"   - Top 5 phrases:")
    for _, row in csv_df.head(5).iterrows():
        print(f"      • '{row['phrase']}' (count: {row['count']})")
    print(f"💾 Saved to: {output_file}")

# ------------------------------------------------------------------
# Combine all results
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print("📊 COMBINING ALL RESULTS")
print(f"{'='*60}")

if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Aggregate across all CSVs (sum counts for same phrase)
    aggregated = combined_df.groupby("phrase").agg({
        "count": "sum",
        "csv_source": lambda x: ", ".join(sorted(set(x)))
    }).reset_index()
    
    aggregated["pos"] = "NOUN_PHRASE"
    aggregated = aggregated.sort_values("count", ascending=False).reset_index(drop=True)
    
    # Save combined results
    combined_file = os.path.join(OUTPUT_DIR, "combined_noun_phrases_all.csv")
    aggregated.to_csv(combined_file, index=False)
    
    print(f"\n✅ Combined results saved to: {combined_file}")
    print(f"📈 Total unique noun phrases across all CSVs: {len(aggregated):,}")
    print(f"📈 Total phrase occurrences across all CSVs: {aggregated['count'].sum():,}")
    
    # Top 10 overall
    print(f"\n🔝 Top 10 most frequent noun phrases (overall):")
    for i, row in aggregated.head(10).iterrows():
        print(f"   {i+1}. '{row['phrase']}' (count: {row['count']}, sources: {row['csv_source']})")

# ------------------------------------------------------------------
# Print detailed summary table
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print("📋 DETAILED SUMMARY TABLE")
print(f"{'='*60}")

summary_df = pd.DataFrame(overall_stats)
print(summary_df.to_string(index=False))

# Save summary stats
summary_file = os.path.join(OUTPUT_DIR, "processing_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"\n💾 Summary stats saved to: {summary_file}")

print(f"\n{'='*60}")
print("🎉 PROCESSING COMPLETE")
print(f"{'='*60}")
