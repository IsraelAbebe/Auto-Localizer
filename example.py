from auto_localizer.localize import add_modified_native, LANGUAGES

# Example data structure (assuming 'eng_preproceed' and native language codes are keys)
example_data = {
    "eng_preproceed": "Janet's ducks lay 16 eggs per day.",
    "fra": "Les canards de Janet pondent 16 œufs par jour.",
    "amh": "የጃኔት ዳክዬዎች በቀን 16 እንቁላሎችን ይጥላሉ።",
    # ... other languages as needed
}

# Process the example for French and Amharic
# Ensure GOOGLE_API_KEY environment variable is set before running
# export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
localized_example = add_modified_native(example_data, languages=["fra", "amh"])

print("Original Example Data:")
print(example_data)
print("\nLocalized Example Data:")
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
