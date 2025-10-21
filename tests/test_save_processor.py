import tempfile
from transformers import WhisperFeatureExtractor, AutoProcessor

from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer

# Assuming the setup from your code
test_beatmap_tokenizer_config = {
    "event_types": ["hitcircle", "slider", "spinner"],
}
test_metadata_tokenizer_config = {
    "modes": ["osu", "taiko", "fruits", "mania"],
    "mappers": ["OliBomby", "Cookiezi", "peppy", "Xenon"],
}
processor = CM3PProcessor(
    WhisperFeatureExtractor(),
    CM3PBeatmapParser(),
    CM3PBeatmapTokenizer(
        vocab_init=test_beatmap_tokenizer_config,
        max_time=8000,
    ),
    CM3PMetadataTokenizer(
        vocab_init=test_metadata_tokenizer_config,
        min_year=2014,
    ),
)

# Use a temporary directory to save and load the processor
with tempfile.TemporaryDirectory() as temp_dir:
    # Save the processor to the temporary directory
    processor.save_pretrained(temp_dir)
    print(f"Processor saved to temporary directory: {temp_dir}")

    # Load the processor from the directory using AutoProcessor
    loaded_processor = AutoProcessor.from_pretrained(temp_dir)
    print("Processor loaded successfully from temporary directory.")

# You can verify that the loaded processor is of the correct type
assert isinstance(loaded_processor, CM3PProcessor)
print(f"Loaded processor type: {type(loaded_processor)}")

assert loaded_processor.to_dict() == processor.to_dict(), "Loaded processor does not match the original processor."
assert loaded_processor.beatmap_tokenizer.max_time == processor.beatmap_tokenizer.max_time, "Max time of loaded processor does not match the original."
assert loaded_processor.beatmap_tokenizer.get_vocab() == processor.beatmap_tokenizer.get_vocab(), "Vocab of loaded processor does not match the original."
assert loaded_processor.metadata_tokenizer.min_year == processor.metadata_tokenizer.min_year, "Min year of loaded processor does not match the original."

