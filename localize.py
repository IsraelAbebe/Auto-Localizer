from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
import random
from datasets import load_dataset, Dataset
import re
import spacy
import os
import google.generativeai as genai
from pydantic import BaseModel
from functools import partial
import difflib
import time
import inspect
import sys

class GeminiRetryableError(Exception):
    """Custom exception for retryable Gemini API errors."""
    pass


os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY")) # Configure the client globally

gemini_client = genai.GenerativeModel("gemini-1.5-pro") # Renamed from 'client' to avoid conflict with `client` in the main script for NER

nlp = spacy.load("en_core_web_sm")

def app_log(message: str, level: str = "INFO"):
    """Prints a log message with filename and line number."""
    frame = inspect.currentframe().f_back
    lineno = frame.f_lineno
    filename = os.path.basename(frame.f_code.co_filename)
    print(f"({filename}:{lineno}) [{level}] {message}")

CATEGORIES = ["person_names", "animal", "food", "time", "organization_name", "currency_symbol_and_name"]
LANGUAGES = ['twi', 'amh' , 'ewe', 'fra', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'wol', 'xho', 'yor', 'zul']
LANGUAGE_NAME_MAP = {
    "amh": "Amharic",
    "ewe": "Ewe",
    "fra": "French",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lin": "Lingala",
    "lug": "Luganda",
    "orm": "Oromo",
    "sna": "Shona",
    "sot": "Southern Sotho",
    "swa": "Swahili",
    "wol": "Wolof",
    "twi":"Twi",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu"
}

PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs"
}

BLACKLIST_SUBSTRING = ["%"]

CURRENCY_SYMBOL_TO_NAME = {
    "$": "dollar",
    "€": "euro",
    "₦": "naira",
    "CFA franc": "CFA franc",
    "Shilingi": "shilingi",
    "Birr": "birr",
    "Loti": "loti",
    "Cedi": "cedi",
    "Rand": "rand"
}

class OutputStruct(BaseModel):
    """
    Structured output for classifying words into categories.
    """
    person_names: list[str]
    currency_symbol_and_name: list[str]
    organization_name: list[str]

    def toJSON(self):
        return {
            "person_names": self.person_names,
            "currency_symbol_and_name": self.currency_symbol_and_name,
            "organization_name":self.organization_name
        }


_gemini_classify_cache = {}

def normalize_entities(entities: list[str]) -> list[str]:
    """
    Normalizes a list of entities to handle possessives and substrings.
    - Removes possessives (e.g., "John's" becomes "John").
    - If one entity is a substring of another, keeps the longer one 
      (e.g., ["John", "John's Farm"] -> ["John's Farm"]).
    """
    app_log(f"Input: {entities}", level="DEBUG")
    if not entities or len(entities) <= 1:
        app_log(f"Output (early exit): {entities}", level="DEBUG")
        return entities

    # Step 1: Handle possessives, ensuring the base form is stored.
    best_forms = {}
    for entity in entities:
        # Get the base form (e.g., "Janet's" -> "Janet") but preserve capitalization
        base_form = re.sub(r"['’]s$", "", entity, flags=re.IGNORECASE)
        # Use a lowercased version for consistent keying
        base_form_lower = base_form.lower()

        if base_form_lower not in best_forms:
            best_forms[base_form_lower] = base_form
        # No else needed: the first form encountered (e.g., "Janet" or "Janet's")
        # determines the capitalization of the stored base form.
    
    clean_entities = list(best_forms.values())
    app_log(f"After possessive handling: {clean_entities}", level="DEBUG")

    if len(clean_entities) <= 1:
        app_log(f"Output (after possessive): {clean_entities}", level="DEBUG")
        return clean_entities

    # Step 2: Handle substrings. Remove any entity that is a substring of another.
    entities_to_remove = set()
    for entity_a in clean_entities:
        for entity_b in clean_entities:
            # Don't compare an entity to itself
            if entity_a == entity_b:
                continue
            
            # If entity_a is a substring of entity_b, mark entity_a for removal.
            # We use lower() for a case-insensitive comparison.
            if entity_a.lower() in entity_b.lower():
                entities_to_remove.add(entity_a)

    # Return the list of entities that were not marked for removal
    final_entities = [e for e in clean_entities if e not in entities_to_remove]
    app_log(f"Final output: {final_entities}", level="DEBUG")
    
    return final_entities


def gemini_classify_api(word_list: list[str]) -> dict:
    """
    Classifies a list of words into predefined categories using the Gemini API.
    Returns the classification as a dictionary.
    """
    cache_key = tuple(word_list)
    if cache_key in _gemini_classify_cache:
        return _gemini_classify_cache[cache_key]

    prompt = f"""
                You are a language parser. Your task is to classify the given words into the following categories.  categories: {', '.join(CATEGORIES)}. Use only the words provided and strictly follow the output format as a JSON object with lists for each category.

                Each category is described below:

                1. **person_names** – Human first names or full names this should not include `s or anny additons just extract the names.  
                - Examples: `"Alice"`, `"John Doe"`, `"Fatima"`.
                - DONT INCLUDE PRONOUNS: `'he'` `'she'` `'it'` `'her'` 

                2. **currency_symbol_and_name** – Currency-related items, including:
                - **Symbols**: e.g., `"$"`, `"€"`, `"₦"`, `"£"`
                - **Currency words**: e.g., `"dollar"`, `"euro"`, `"naira"`
                - **Abbreviations**: e.g., `"USD"`, `"EUR"`, `"NGN"`, `"GBP"`
                - DONT INCLUDE PRONOUNS: `%` `percent` `pound` since its confusing with weight
               
                3. **organization_name** – Names of institutions, businesses, or other organizations.  
                - Examples:  `"World Health Organization"`, `"Harvard University"`

                 Instructions:
                - If a category has no matching items, return an empty list `[]` for it.
                - Do not include explanations or extra text.
                - Return only valid items under each category.
                - Preserve the exact format and spelling of the input words.


                Words: {', '.join(word_list)}"""

    # For Gemini, system instructions for schema are handled by response_schema
    response = None
    for attempt in range(3): # Retry up to 3 times
        try:
            response = gemini_client.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": OutputStruct, # Directly use the Pydantic model for schema
                },
            )
            break # Success
        except Exception as e:
            if "500" in str(e) and attempt < 2:
                app_log(f"Gemini API 500 error in classification. Retrying in 2s... (Attempt {attempt+1}/3)", level="WARN")
                time.sleep(3)
            else:
                app_log(f"Gemini API call failed after multiple retries: {e}", level="ERROR")
                raise e # Re-raise the exception if it's not a 500 or retries are exhausted
    
    if response is None:
        # This case handles where the loop finishes without a successful response or exception
        raise Exception("Failed to get a response from Gemini API for classification.")
    
    # Gemini automatically parses into the Pydantic object if response_schema is used
    # app_log(response)
    classified_words_str: str = response.candidates[0].content.parts[0].text
    
    try:
        # The response is a JSON string, so parse it.
        data = json.loads(classified_words_str)
        
        # Filter out pronouns from person_names before normalization
        if 'person_names' in data and isinstance(data['person_names'], list):
            data['person_names'] = [name for name in data['person_names'] if name.lower() not in PRONOUNS]

        # Normalize each list in the dictionary for relevant keys
        if 'person_names' in data and isinstance(data['person_names'], list):
            data['person_names'] = normalize_entities(data['person_names'])
        if 'organization_name' in data and isinstance(data['organization_name'], list):
            data['organization_name'] = normalize_entities(data['organization_name'])
        
    except (json.JSONDecodeError, TypeError):
        # Fallback if parsing fails, though it shouldn't with Gemini's schema enforcement
        app_log(f"Failed to parse Gemini JSON: {classified_words_str}", level="ERROR")
        data = {"person_names": [], "currency_symbol_and_name": [], "organization_name": []}

    app_log(f"Classification result: {data}", level="DEBUG")
    _gemini_classify_cache[cache_key] = data
    return data

# --- Gemini Version of translate_with_substitution ---
_gemini_translate_cache = {}

def gemini_translate_with_substitution(original_eng: str, original_native: str, modified_eng: str, native_lang: str, replacements: dict, retries: int = 3, similarity_threshold: float = 0.7) -> str:
    """
    Translates/modifies a native sentence based on changes made in an English sentence using Gemini.
    """
    cache_key = (original_eng, original_native, modified_eng, native_lang)
    if cache_key in _gemini_translate_cache:
        return _gemini_translate_cache[cache_key]

    prompt = f"""You are an expert linguistic assistant. Your task is to edit a sentence in a native language to match a change made in its English parallel.

                Here is the context:
                1.  **Original English:** The original sentence.
                2.  **Original Native:** The original translation of the English sentence in {native_lang}.
                3.  **Modified English:** The English sentence has been edited. One or more words have been replaced.

                Your goal is to produce a **Modified Native** sentence by applying the *exact same replacement* to the **Original Native** sentence.

                **Crucial Instructions:**
                -   **DO NOT re-translate the entire sentence.** Only replace the specific words that were changed in the English version.
                -   Preserve the original grammar and structure of the native sentence as much as possible.
                -   Ensure the final "Modified Native" sentence is natural and grammatically correct in {native_lang}.
                -   Respond with ONLY the "Modified Native" sentence and nothing else.

                **Example:**
                -   Original English: Janet's ducks lay 16 eggs per day.
                -   Original Native (French): Les canards de Janet pondent 16 œufs par jour.
                -   Modified English: Andrea's ducks lay 16 eggs per day.
                -   Modified Native (French): Les canards d'Andrea pondent 16 œufs par jour.

                **Your Task:**

                Original English:
                {original_eng}

                Original Native ({native_lang}):
                {original_native}

                Modified English:
                {modified_eng}

                Modified Native ({native_lang}):
                """

    for attempt in range(retries):
        response = None
        for api_attempt in range(3): # Retry up to 3 times for API errors
            try:
                response = gemini_client.generate_content(
                    contents=prompt,
                    generation_config={"temperature": 0.0}
                )
                break # Success
            except Exception as e:
                if "500" in str(e) and api_attempt < 2:
                    app_log(f"Gemini API 500 error in translation. Retrying in 2s... (Attempt {api_attempt+1}/3)", level="WARN")
                    time.sleep(3)
                else:
                    app_log(f"Gemini API call failed after multiple retries: {e}", level="ERROR")
                    raise e # Re-raise the exception

        if response is None:
            raise Exception("Failed to get a response from Gemini API for translation.")

        translation = response.text.strip()

        # Check for the specific error message in the response text
        if "ERROR: 500 An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting" in translation:
            raise GeminiRetryableError(f"Gemini API returned retryable error: {translation}")

        # Validate if all replacement values are in the translation
        if replacements:
            missing_words = [val for val in replacements.values() if val.lower() not in translation.lower()]
            if not missing_words:
                # Similarity check
                original_text_for_comparison = original_native
                translated_text_for_comparison = translation
                for val in replacements.values():
                    original_text_for_comparison = original_text_for_comparison.replace(val, "")
                    translated_text_for_comparison = translated_text_for_comparison.replace(val, "")
                
                similarity = difflib.SequenceMatcher(None, original_text_for_comparison, translated_text_for_comparison).ratio()

                if similarity >= similarity_threshold:
                    _gemini_translate_cache[cache_key] = translation
                    return translation
                else:
                    prompt += f"\n\nYour previous translation was too different from the original (similarity: {similarity:.2f}). Please stick closer to the original text, only changing the necessary words."

            else:
                prompt += f"\n\nYour previous translation was missing the following words: {', '.join(missing_words)}. Please ensure they are included in the corrected translation."
        else:
            # If there are no replacements, no need to validate
            _gemini_translate_cache[cache_key] = translation
            return translation

    # Return the last attempt if all retries fail
    _gemini_translate_cache[cache_key] = translation
    return translation


def get_nouns(text):
    doc = nlp(text)
    seen = set()
    unique_nouns = []
    for token in doc:
        if token.pos_ == "NOUN":
            lemma = token.lemma_.lower()
            if lemma not in seen:
                seen.add(lemma)
                unique_nouns.append(lemma)
    return unique_nouns

def get_random_entity(json_path: str, language_code: str, entity_type: str):
    app_log(f"get_random_entity {json_path,language_code,entity_type}")
    # Normalize entity_type input to lower-case and standard keys
    entity_type = entity_type.lower()
    valid_types = {
            "person_names": "person_names",
            "currency_symbol_and_name": "currency_symbol_and_name",
            "organization_name": "organization_name"
        }

    if entity_type not in valid_types:
        raise ValueError(f"Invalid entity type. Choose from: {list(valid_types.keys())}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if language_code not in data:
        raise ValueError(f"Language '{{language_code}}' not found in data.")

    entities = data[language_code].get(entity_type, [])
    if not entities:
        raise ValueError(f"No entities found for type '{{entity_type}}' in language '{{language_code}}'.")

    return random.choice(entities)

def generate_replacement_dict(entity_dict: dict, language_code: str, selected_entity: list[str] = [], json_path: str = "entities.json"):
    replacement_dict = {}
    app_log(f"entity_dict-->  {entity_dict} - {selected_entity}", level="DEBUG")

    # Load all possible entities for the given language once to get the pools of replacements.
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_entities_for_lang = json.load(f).get(language_code, {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        app_log(f"Could not load or parse entities file '{json_path}': {e}", level="ERROR")
        return {} # Return empty dict if entities can't be loaded

    filtered_entity_list = {k: entity_dict[k] for k in selected_entity if k in entity_dict}
    app_log(f"filtered_entity_list-->  {filtered_entity_list}", level="DEBUG")

    for entity_type, entity_list_to_replace in filtered_entity_list.items():
        if not entity_list_to_replace:
            continue

        # Get the pool of available replacements for this type and shuffle for randomness.
        available_replacements = all_entities_for_lang.get(entity_type, []).copy()
        random.shuffle(available_replacements)

        if not available_replacements:
            app_log(f"No replacement entities found for type '{entity_type}' in language '{language_code}'.", level="WARN")
            continue

        for entity_to_replace in entity_list_to_replace:
            if entity_to_replace.lower() in PRONOUNS:
                app_log(f"Skipping pronoun: {entity_to_replace}", level="DEBUG")
                continue

            # If the entity type is currency and there's only one replacement, reuse it.
            if entity_type == 'currency_symbol_and_name' and len(all_entities_for_lang.get(entity_type, [])) == 1:
                translated = all_entities_for_lang[entity_type][0]
            else:
                # Check if we have any unique replacements left in the pool.
                if not available_replacements:
                    app_log(f"Ran out of unique replacements for type '{entity_type}'. Cannot replace '{entity_to_replace}'.", level="WARN")
                    break  # Stop replacing for this entity type to maintain uniqueness.

                # Pop a unique replacement from the shuffled list.
                translated = available_replacements.pop()
            
            replacement_dict[entity_to_replace] = translated
    
    # Manual filtering for blacklisted substrings
    final_replacements = {}
    for key, value in replacement_dict.items():
        is_blacklisted = False
        for term in BLACKLIST_SUBSTRING:
            if term in key.lower():
                app_log(f"Skipping '{key}' because it contains blacklisted term '{term}'.", level="DEBUG")
                is_blacklisted = True
                break
        if not is_blacklisted:
            final_replacements[key] = value

    # Expand currency replacements
    expanded_replacements = final_replacements.copy()
    for key, value in final_replacements.items():
        if key in CURRENCY_SYMBOL_TO_NAME:
            currency_name = CURRENCY_SYMBOL_TO_NAME[key]
            if currency_name not in expanded_replacements:
                expanded_replacements[currency_name] = value

    app_log(f"Final replacement dict: {expanded_replacements}", level="DEBUG")
    return expanded_replacements

import re

def replace_from_dict(text: str, replacements: dict) -> str:
    app_log(f"Initial text: {text}", level="DEBUG")
    app_log(f"Replacements: {replacements}", level="DEBUG")

    text = text.lower()

    # Sort keys by length (longest first to avoid partial replacements)
    sorted_keys = sorted(replacements.keys(), key=len, reverse=True)

    for key in sorted_keys:
        # Skip if key is a pronoun
        if key.lower() in PRONOUNS:
            continue

        replacement = replacements[key]

        # Check if the key contains any alphabetic character to treat it as a word/name.
        # This is more robust than key.isalpha() as it handles names with apostrophes (e.g., "Janet's").
        if any(c.isalpha() for c in key):
            # Strip possessive from the key to create a clean base for the regex.
            # This makes the matching robust, whether the key is "Janet" or "Janet's".
            base_key = re.sub(r"['’]s$", "", key)
            
            # Pattern to match the base word, optionally followed by a possessive 's or ’s.
            pattern = re.compile(
                rf'\b{re.escape(base_key)}(?:’s|\'s)?\b',
                flags=re.IGNORECASE | re.UNICODE
            )
        else:  # It's a symbol (like '$')
            # This pattern looks for the symbol, optional space, and an adjoining digit.
            pattern = re.compile(rf'{re.escape(key)}\s?(?=\d)', flags=re.UNICODE)

        text = pattern.sub(replacement + ' ', text)

    app_log(f"Final text: {text}", level="DEBUG")
    return text

def ner_to_entity_dict(ner_results: list[dict], original_text: str = None) -> dict:
    """
    Extracts entities from NER results and categorizes them, including currency_symbol_and_name via regex.
    Note: This function is present in the original code but `extract_ner_entities`
    is commented out. If you intend to use it, ensure `extract_ner_entities`
    is active and its dependencies are met.
    """
    # Mapping from NER tags to readable categories
    tag_mapping = {
        "B-PER": "person", "I-PER": "person",
        "B-LOC": "location", "I-LOC": "location",
        "B-ORG": "organization_name", "I-ORG": "organization_name",
        "B-MISC": "misc", "I-MISC": "misc"
    }

    entity_dict = {}

    # Step 1: Use NER output
    for item in ner_results:
        word = item["word"]
        tag = item["entity"]

        # Check tag for other types
        entity_type = tag_mapping.get(tag)
        if entity_type:
            entity_dict.setdefault(entity_type, []).append(word)

    return entity_dict

def extract_ner_entities(text: str, model_name: str = " "):
    """
    Extracts Named Entities using a Hugging Face Transformers pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    return ner_pipeline(text)


def add_modified_native(example: dict, languages: list[str] = LANGUAGES, verbose: bool = False) -> dict:
    app_log(f"Processing example. Available languages for this entry: {list(example.keys())}")

    processed_any_language = False
    for lang_code in languages:
        if lang_code not in example or not example[lang_code]:
            app_log(f"Skipping language '{lang_code}': Not present or empty in example.", level="WARNING")
            continue

        processed_any_language = True
        app_log(f"--- Processing language: {LANGUAGE_NAME_MAP.get(lang_code, lang_code)} ({lang_code}) ---")

        original_eng = example.get("eng_preproceed", "").lower()
        # original_eng = re.sub(r'(\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\d+\s*\$)', r'$ \1', original_eng)
        original_native = example.get(lang_code, "").lower()

        # Create a copy of the original English text
        modified_eng = original_eng[:]

        if not original_eng or not original_native:
            app_log(f"Skipping language '{lang_code}': Original English or Native text is empty.", level="WARNING")
            example[f"{lang_code}_loc"] = "" # Ensure the column exists even if skipped
            example[f"{lang_code}_replacements_made"] = "NONE (empty original text)"
            continue

        try:
            app_log(f"Original English: {original_eng}")
            app_log(f"Original Native ({LANGUAGE_NAME_MAP.get(lang_code, lang_code)}): {original_native}")

            nouns = get_nouns(original_eng)
            app_log(f"Extracted nouns: {nouns}")

            if not nouns:
                app_log("No nouns extracted, skipping entity processing for this language.")
                example[f"{lang_code}_loc"] = original_native # Keep original if no nouns to process
                example[f"{lang_code}_replacements_made"] = "NONE (no nouns)"
                continue

            # --- Use Gemini for classification ---
            classified_entities = gemini_classify_api(original_eng.split(" "))
            app_log(f"Entity classification result: {json.dumps(classified_entities, indent=2)}")

            selected_entity_types = ["person_names", "currency_symbol_and_name","organization_name"] # Customize as needed
            app_log(f"Selected entity types for replacement: {selected_entity_types}")

            replacements = generate_replacement_dict(classified_entities, lang_code, selected_entity=selected_entity_types, json_path="entities.json") # Ensure json_path is passed if not default
            app_log(f"Generated replacement dictionary: {json.dumps(replacements, ensure_ascii=False)}")


            modified_eng = replace_from_dict(original_eng.lower(), replacements)
            app_log(f"Modified English sentence: {modified_eng}")
            # if not replacements:
            #     app_log("No replacements generated, using original native text.")
            #     modified_native = original_native
            #     example[f"{lang_code}_loc"] = original_eng
            #     example[f"{lang_code}_replacements_made"] = None
            # else:
            #     # Preprocess text to add spaces around currency_symbol_and_name symbols and numbers
            #     #FIXTHIS
            #     # 

            #     # IMPORTANT: The original code converts original_eng to lowercase before replacement.
            #     # Ensure this is consistent with how the translation model expects input.
            #     modified_eng = replace_from_dict(original_eng.lower(), replacements)
            #     app_log(f"Modified English sentence: {modified_eng}")

            #     if modified_eng:
            #         example[f"{lang_code}_loc"] = modified_eng
            #         example[f"{lang_code}_replacements_made"] = json.dumps(replacements) if replacements else "NONE" # Store as JSON string
            #     else:
            #         example[f"{lang_code}_loc"] = original_eng
            #         example[f"{lang_code}_replacements_made"] = None
                

            

            if original_eng.lower() != modified_eng.lower(): # Compare lowercased versions for changes
                max_retries = 5
                for retry_attempt in range(max_retries):
                    try:
                        modified_native = gemini_translate_with_substitution(
                            original_eng.lower(), # Pass lowercased original as well for consistency
                            original_native.lower(), # Pass lowercased original native for consistency
                            modified_eng.lower(),
                            native_lang=LANGUAGE_NAME_MAP.get(lang_code, lang_code),
                            replacements=replacements
                        )
                        app_log(f"Modified native sentence: {modified_native}")
                        break # Break out of retry loop on success
                    except GeminiRetryableError as e:
                        app_log(f"Retryable Gemini translation error for {lang_code} (Attempt {retry_attempt + 1}/{max_retries}): {e}", level="WARNING")
                        if retry_attempt < max_retries - 1:
                            time.sleep(3 ** retry_attempt) # Exponential backoff
                        else:
                            sys.exit() # Exit retry loop
                    except Exception as e:
                        app_log(f"Non-retryable error during Gemini translation for {lang_code}: {e}", level="ERROR")
                        sys.exit() # Exit retry loop # Exit retry loop
                else: # This else block executes if the loop completes without a 'break'
                    # This means all retries failed and the last attempt was a non-retryable error or max retries reached
                    pass # The error message is already set in the except blocks above
            else:
                app_log("Modified English is same as original English (after lowercasing), no translation needed.")
                modified_native = original_native

            example[f"{lang_code}_loc"] = modified_native
            # example[f"eng_{lang_code}_loc"] = modified_eng
            if f"{lang_code}_replacements_made" not in example: # Only set if not already set by retry logic
                example[f"{lang_code}_replacements_made"] = json.dumps(replacements) if replacements else "NONE" # Store as JSON string

        except Exception as e:
            app_log(f"Error processing language {lang_code} for example: {e}", level="ERROR")
            sys.exit() # Exit retry loop

    if not processed_any_language:
        app_log("No languages were processed for this example.", level="WARNING")
    return example

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Localize dataset entries.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run the localization generation process."
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run in experiment mode, processing a limited number of samples."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process. Used with --experiment or to limit --generate."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="israel/ldYgx5w24IGOdMf",
        help="Name of the dataset on Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="default",
        help="Configuration of the dataset (e.g., 'afrimgsm')."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'test', 'train')."
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default="israel/loc",
        help="Name for the output dataset on Hugging Face Hub."
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="3_eng_our.json",
        help="Name of the checkpoint file to save progress.")
        # default="localization_checkpoint.json",
    # )


    args = parser.parse_args()

    # --- Configure Gemini API key ---
    if os.getenv("GOOGLE_API_KEY") is None:
        app_log("GOOGLE_API_KEY environment variable not set. Please set it or hardcode your key for testing.", level="WARNING")
    else:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    if not args.generate and not args.experiment:
        app_log("Neither --generate nor --experiment flag was set. Exiting. Use --help for options.")
    else:
        app_log("Starting script...")

        # --- Checkpointing Logic ---
        processed_examples = []
        start_index = 0
        if os.path.exists(args.checkpoint_file):
            app_log(f"Found checkpoint file: {args.checkpoint_file}. Resuming progress.")
            with open(args.checkpoint_file, "r", encoding="utf-8") as f:
                try:
                    processed_examples = json.load(f)
                    start_index = len(processed_examples)
                    app_log(f"Resuming from index {start_index}.")
                except json.JSONDecodeError:
                    app_log(f"Checkpoint file {args.checkpoint_file} is corrupted. Starting from scratch.", level="WARNING")
                    processed_examples = []
                    start_index = 0
        
        dataset_to_process = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        app_log(f"Loaded dataset '{args.dataset_name}' (config: {args.dataset_config}, split: {args.split}). Total samples: {len(dataset_to_process)}")

        num_to_process = args.num_samples
        if args.experiment and num_to_process is None:
            num_to_process = 5  # Default for experiment mode
        
        # Determine the actual slice of the dataset to process
        if num_to_process is not None:
            dataset_slice = dataset_to_process.select(range(min(num_to_process, len(dataset_to_process))))
        else:
            dataset_slice = dataset_to_process
        
        app_log(f"Target number of samples to process: {len(dataset_slice)}")

        if len(dataset_slice) == 0:
            app_log("No samples to process. Exiting.", level="WARNING")
        elif start_index >= len(dataset_slice):
            app_log("All target samples already processed based on checkpoint file.")
            updated_dataset = Dataset.from_list(processed_examples)
        else:
            try:
                for i in range(start_index, len(dataset_slice)):
                    example = dataset_slice[i]
                    app_log(f"--- Processing example {i+1}/{len(dataset_slice)} ---")
                    
                    processed_example = add_modified_native(example, languages=LANGUAGES, verbose=args.verbose)
                    processed_examples.append(processed_example)
                    
                    with open(args.checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(processed_examples, f, indent=2, ensure_ascii=False)

                app_log("All examples processed successfully.")
                
                updated_dataset = Dataset.from_list(processed_examples)
                
            except KeyboardInterrupt:
                app_log("\nProcess interrupted by user. Progress has been saved.")
                exit() 
            except Exception as e:
                app_log(f"\nAn error occurred: {e}. Progress has been saved.", level="ERROR")
                exit() 

        if (args.generate or args.experiment) and 'updated_dataset' in locals():
            try:
                app_log(f"Pushing updated dataset to Hugging Face Hub: {args.output_dataset_name}")
                updated_dataset.push_to_hub(args.output_dataset_name)
                app_log("Successfully pushed to hub.")
                
            except Exception as e:
                app_log(f"Error pushing dataset to hub: {e}", level="ERROR")

        app_log("Script finished.")
