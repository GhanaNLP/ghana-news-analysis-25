# ------------------------------------------------------------
# Auto-install missing requirements
# ------------------------------------------------------------
import sys, subprocess, importlib

pkgs = {
    "pandas": "pandas>=1.5.0",
    "openai": "openai>=1.0.0",
    "tqdm": "tqdm>=4.64.0"
}
for import_name, pip_spec in pkgs.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"📦 {import_name} not found → installing {pip_spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])

# ------------------------------------------------------------
# Normal imports
# ------------------------------------------------------------
import pandas as pd
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError
import json
from tqdm import tqdm
import time
from collections import defaultdict
import os

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
NVIDIA_API_KEY = "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"  # ← put your NVIDIA API key here
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
MODEL = "moonshotai/kimi-k2-instruct-0905" 
BATCH_SIZE = 300 
RESUME_STATE_FILE = "classification_resume_state.json"

# -----------------------------------------------------------
# RETRY CONFIG
# -----------------------------------------------------------
MAX_NETWORK_RETRIES = 100
MAX_CONTENT_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 200
BACKOFF_FACTOR = 2

# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------
def save_resume_state(batch_num, word_classifications, total_batches, input_file, output_file, text_column):
    state = {
        "batch_num": batch_num,
        "word_classifications": word_classifications,
        "total_batches": total_batches,
        "input_file": input_file,
        "output_file": output_file,
        "text_column": text_column,
        "timestamp": time.time(),
        "batch_size": BATCH_SIZE,
    }
    with open(RESUME_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved resume state (completed batch {batch_num}/{total_batches})")

def load_resume_state():
    if not os.path.exists(RESUME_STATE_FILE):
        return None
    try:
        with open(RESUME_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Error loading resume state: {e}")
        return None

def clear_resume_state():
    if os.path.exists(RESUME_STATE_FILE):
        os.remove(RESUME_STATE_FILE)
    print("  🗑️  Cleared resume state")

def calculate_retry_delay(attempt, is_network_error=False):
    if is_network_error:
        delay = min(INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** (attempt - 1)), MAX_RETRY_DELAY)
    else:
        delay = min(INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** (attempt - 1)), 30)
    jitter = delay * 0.1
    delay += (jitter * (2 * (time.time() % 1) - 1))
    return max(1, delay) 

def safe_sleep(seconds, interrupt_check_interval=5):
    end_time = time.time() + seconds
    while time.time() < end_time:
        remaining = end_time - time.time()
        sleep_time = min(interrupt_check_interval, remaining)
        if sleep_time <= 0:
            break
        time.sleep(sleep_time)

def classify_batch_entity_type(words):
    """
    Refined to capture only HIGH-SPECIFICITY, INDUSTRY-RELATED ACTIVITIES.
    Rejects vague organizational nouns (Action, Service, Operations).
    """
    n = len(words)
    word_list = "\n".join([f"{i+1}. {w}" for i, w in enumerate(words)])

    prompt = f"""
Classify each word into: KEYWORD or NON-KEYWORD.

STRICT SPECIFICITY RULES:
1. KEYWORD (Specific development related Activity):
   - Must be a specific development activity, not a general category actvitity

2. NON-KEYWORD (Vague, Organizational, or Abstract):
   - GENERAL ACTIVITIES: health
   - VAGUE ACTIONS: action, service, operations, finance, process, management, effort, support, work.
   - SOCIAL/GENERIC EVENTS: interview, meeting, fight, visit, incident, issue.
   - BROAD CONCEPTS: corruption, integrity, confidence, progress, development, culture.
   - DOCUMENTS/ENTITIES: report, article, link, OSP, ministry, budget.

Words to classify:
{word_list}

Respond ONLY with a JSON array of objects.
Example: [{{"word": "welding", "label": "KEYWORD"}}, {{"word": "service", "label": "NON-KEYWORD"}}]
"""

    # ... (Keep the rest of your retry and JSON repair logic from the previous response)
    network_attempt = 0
    content_attempt = 0

    while True:
        try:
            network_attempt += 1
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.95,
                max_tokens=n * 50,
                stream=False,
                timeout=120
            )

            content = completion.choices[0].message.content.strip()

            # Clean JSON block if model includes markdown
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'): content = content[4:]
                content = content.split('```')[0].strip()

            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found")

            raw_data = json.loads(content[start_idx:end_idx])

            # Validation and Repair (Fixes the 'str' object has no attribute 'get' error)
            final_results = []
            for i, item in enumerate(raw_data):
                if i >= n: break 
                
                # If item is just a string (the label), reconstruct the dict
                if isinstance(item, str):
                    label = "KEYWORD" if "KEYWORD" in item.upper() and "NON" not in item.upper() else "NON-KEYWORD"
                    final_results.append({"word": words[i], "label": label})
                # If item is a dict, validate the label
                elif isinstance(item, dict):
                    lbl = str(item.get("label", "NON-KEYWORD")).upper()
                    item["label"] = "KEYWORD" if "KEYWORD" in lbl and "NON" not in lbl else "NON-KEYWORD"
                    final_results.append(item)
                else:
                    final_results.append({"word": words[i], "label": "NON-KEYWORD"})

            # Ensure we return exactly N items
            while len(final_results) < n:
                final_results.append({"word": words[len(final_results)], "label": "NON-KEYWORD"})
                
            return final_results

        except (APITimeoutError, APIConnectionError, APIError) as e:
            delay = calculate_retry_delay(network_attempt, is_network_error=True)
            print(f"  🔄 Network Error. Retrying in {delay:.1f}s...")
            safe_sleep(delay)
        except (json.JSONDecodeError, ValueError) as e:
            content_attempt += 1
            if content_attempt >= MAX_CONTENT_RETRIES: return None
            delay = calculate_retry_delay(content_attempt)
            print(f"  🔄 Formatting Error: {e}. Retrying...")
            safe_sleep(delay)

# -----------------------------------------------------------
# helper – build unique-word list + reverse index
# -----------------------------------------------------------
def get_unique_words_with_mapping(df, text_column):
    texts = df[text_column].astype(str).str.strip().tolist()
    word_to_indices = defaultdict(list)
    for idx, word in enumerate(texts):
        word_to_indices[word].append(idx)
    unique_words = list(word_to_indices.keys())
    print(f"Total rows: {len(texts)}")
    print(f"Unique words: {len(unique_words)}")
    print(f"Duplicates removed: {len(texts) - len(unique_words)} "
          f"({(1 - len(unique_words)/len(texts))*100:.1f}% reduction)")
    return unique_words, word_to_indices

# -----------------------------------------------------------
# 2.  BATCH PROCESSOR
# -----------------------------------------------------------
def process_csv_in_batches(input_file, output_file, text_column='text',
                           use_thinking=False, filter_keyword_only=False, resume=True):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"\n{'='*60}")
    print(f"Original rows: {len(df)}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Scheme: CONCRETE ACTIVITY (Keyword) vs ABSTRACT/ENTITY (Non-Keyword)")
    print(f"Filter keyword-only: {'Yes' if filter_keyword_only else 'No'}")
    print(f"{'='*60}\n")

    unique_words, word_to_indices = get_unique_words_with_mapping(df, text_column)
    total_batches = (len(unique_words) + BATCH_SIZE - 1) // BATCH_SIZE

    start_batch, word_classifications = 0, {}
    if resume:
        state = load_resume_state()
        if (state and state.get('input_file') == input_file and
            state.get('output_file') == output_file and state.get('batch_size') == BATCH_SIZE):
            start_batch = state['batch_num']
            word_classifications = state['word_classifications']
            print(f"🔄 Resuming from batch {start_batch + 1}/{total_batches}")
        else:
            print("  🆕 Starting fresh (no valid resume state)")

    print(f"\nProcessing {len(unique_words)} unique words in {total_batches} batches...")

    successful_batches = failed_batches = 0
    classify_func = classify_batch_entity_type

    try:
        for i in tqdm(range(start_batch * BATCH_SIZE, len(unique_words), BATCH_SIZE),
                      desc="Processing", initial=start_batch, total=total_batches):

            batch = unique_words[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            classifications = classify_func(batch)

            if classifications:
                for cls in classifications:
                    if isinstance(cls, dict) and 'word' in cls and 'label' in cls:
                        word_classifications[cls['word']] = cls['label']
                successful_batches += 1
            else:
                print(f"  ⚠️  Classifying {len(batch)} words as UNKNOWN after retries")
                for w in batch:
                    word_classifications[w] = 'UNKNOWN'
                failed_batches += 1

            if resume:
                save_resume_state(batch_num, word_classifications,
                                total_batches, input_file, output_file, text_column)

            if i + BATCH_SIZE < len(unique_words):
                time.sleep(0.5)

        if resume:
            clear_resume_state()
            print("  ✅ Processing completed successfully")

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user!")
        print("  💾 Progress saved – resume with the same command")
        return None, None
    except Exception as e:
        print(f"\n\n❌ Error during processing: {e}")
        print("  💾 Progress saved – resume with the same command")
        return None, None

    # apply classifications
    print(f"\n{'='*60}")
    print(f"Applying classifications to all {len(df)} rows...")
    print(f"{'='*60}\n")

    labels = ['UNKNOWN'] * len(df)
    for word, indices in word_to_indices.items():
        label = word_classifications.get(word, 'UNKNOWN')
        for idx in indices:
            labels[idx] = label
    df['classification'] = labels

    if filter_keyword_only:
        out_df = df[df['classification'] == 'KEYWORD'].copy()
    else:
        out_df = df[df['classification'].isin(['KEYWORD', 'NON-KEYWORD'])].copy()
    
    out_df = out_df.drop(columns=['classification'])
    out_df.to_csv(output_file, index=False)

    full_out = output_file.replace('.csv', '_with_labels.csv')
    df.to_csv(full_out, index=False)

    # stats
    keywords = (df['classification'] == 'KEYWORD').sum()
    non_keywords = (df['classification'] == 'NON-KEYWORD').sum()
    unknown = (df['classification'] == 'UNKNOWN').sum()

    print(f"\n{'='*60}\nFINAL RESULTS:\n{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"KEYWORD (Concrete Activities): {keywords}")
    print(f"NON-KEYWORD (Abstract/Entities): {non_keywords}")
    print(f"UNKNOWN: {unknown}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    return out_df, df


# -----------------------------------------------------------
# 3.  TEST + MAIN
# -----------------------------------------------------------
def test_api_connection():
    print("=" * 60)
    print("Testing NVIDIA Build API connection...")
    print("=" * 60 + "\n")
    if NVIDIA_API_KEY == "nvapi-YOUR_API_KEY_HERE":
        print("✗ Error: API key not set!")
        return False
    print(f"Model: {MODEL}\n")
    print("Testing distinction: CONCRETE ACTIVITY vs. ABSTRACT CONCEPT")
    print("----------------------------------------------------------")
    print("- 'farming'     should be KEYWORD     (Concrete action)")
    print("- 'voting'      should be KEYWORD     (Concrete action)")
    print("- 'education'   should be NON-KEYWORD (Abstract system)")
    print("- 'leadership'  should be NON-KEYWORD (Abstract quality)")
    print("- 'development' should be NON-KEYWORD (Abstract concept)")
    print()
    
    test_words = ["farming", "voting", "education", "leadership", "development", "teacher"]
    result = classify_batch_entity_type(test_words)
    if result:
        print("\n✓ API connection successful!\n")
        for item in result:
            sym = "✓" if item['label'] == 'KEYWORD' else "•"
            print(f"  {sym} {item['word']:15} → {item['label']}")
        return True
    else:
        print("\n✗ API connection test failed")
        return False


if __name__ == "__main__":
    if NVIDIA_API_KEY == "nvapi-YOUR_API_KEY_HERE":
        print("\n" + "=" * 60 + "\n⚠️  NVIDIA API KEY REQUIRED\n" + "=" * 60)
    else:
        if test_api_connection():
            print("\n" + "=" * 60 + "\nStarting full CSV processing...\n" + "=" * 60 + "\n")
            time.sleep(2)
            process_csv_in_batches(
                input_file='/home/owusus/Documents/GitHub/ghana-news-analysis/concept_nouns_only.csv',
                output_file='/home/owusus/Documents/GitHub/ghana-news-analysis/concrete_activities_only.csv',
                text_column='phrase',
                use_thinking=False,
                filter_keyword_only=True,
                resume=True
            )
