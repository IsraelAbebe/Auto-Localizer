# Auto-Localizer

`Auto-Localizer` is a Python library designed to automate the localization of text by identifying and replacing specific entities (like person names, currency symbols, and organization names) in English sentences and then applying the corresponding changes to their native language translations. 

paper - [Bridging the Culture Gap: A Framework for LLM-Driven Socio-Cultural Localization of Math Word Problems in Low-Resource Languages](https://www.arxiv.org/abs/2508.14913)

## Features

- **Entity Recognition**: Identifies key entities in English text using a combination of spaCy and Gemini API.
- **Dynamic Replacement**: Replaces identified entities with random alternatives from a predefined list, ensuring variety in localized content.
- **Contextual Translation**: Utilizes the Gemini API to translate/modify native language sentences based on the changes made in the English parallel, preserving grammatical correctness and natural flow.
- **Installable Package**: Easily installable via `pip`, providing both a command-line interface and a Python API.

## Installation

To install `Auto-Localizer`, navigate to the root directory of this project (where `setup.py` is located) and run:

```bash
pip install .
```

This will install the package and its dependencies.

## Usage

### Command-Line Interface

The library provides a command-line tool `localize` for processing datasets.

```bash
localize --help
```

Example usage:

```bash
localize --generate --dataset_name "your_hf_dataset/name" --dataset_config "your_config" --split "test" --output_dataset_name "your_output_dataset/name"
```

### Python API

You can also import and use the `add_modified_native` function directly in your Python scripts.

```python
from auto_localizer.localize import add_modified_native, LANGUAGES

# Example data structure (assuming 'eng_preproceed' and native language codes are keys)
example_data = {
    "eng_preproceed": "Janet's ducks lay 16 eggs per day.",
    "fra": "Les canards de Janet pondent 16 œufs par jour.",
    "amh": "የጃኔት ዳክዬዎች በቀን 16 እንቁላሎችን ይጥላሉ።",
    # ... other languages as needed
}

# Process the example for French and Amharic
localized_example = add_modified_native(example_data, languages=["fra", "amh"])

print(localized_example)
# Expected output (entities will be randomized):
# {
#     'eng_preproceed': "janet's ducks lay 16 eggs per day.",
#     'fra': 'les canards de janet pondent 16 œufs par jour.',
#     'amh': 'የጃኔት ዳክዬዎች በቀን 16 እንቁላሎችን ይጥላሉ።',
#     'fra_loc': 'les canards d\'andrea pondent 16 œufs par jour.',
#     'fra_replacements_made': '{"Janet": "Andrea"}',
#     'amh_loc': 'የአብይ ዳክዬዎች በቀን 16 እንቁላሎችን ይጥላሉ።',
#     'amh_replacements_made': '{"Janet": "አብይ"}'
# }
```

## Configuration

The `GOOGLE_API_KEY` environment variable must be set for the Gemini API calls to work.

```bash
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

The `entities.json` file, located within the `auto_localizer` package, contains the lists of entities used for replacement. You can modify this file to customize the available entities for each language.

## Dependencies

The core dependencies are listed in `setup.py` and will be installed automatically. Key dependencies include:

- `transformers`
- `google-generativeai`
- `spacy`
- `datasets`
- `pydantic`

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues on the GitHub repository.

