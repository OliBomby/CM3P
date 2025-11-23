import tempfile
import pytest
from transformers import WhisperFeatureExtractor, AutoProcessor

from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer

test_beatmap_tokenizer_config = {
    "event_types": ["hitcircle", "slider", "spinner"],
}
test_metadata_tokenizer_config = {
    "modes": ["osu", "taiko", "fruits", "mania"],
    "mappers": ["OliBomby", "Cookiezi", "peppy", "Xenon"],
}


@pytest.fixture(scope="module")
def processor():
    return CM3PProcessor(
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


def test_save_and_load_processor(processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_pretrained(temp_dir)
        loaded_processor = AutoProcessor.from_pretrained(temp_dir)

    assert isinstance(loaded_processor, CM3PProcessor)
    assert loaded_processor.to_dict() == processor.to_dict()
    assert loaded_processor.beatmap_tokenizer.max_time == processor.beatmap_tokenizer.max_time
    assert loaded_processor.beatmap_tokenizer.get_vocab() == processor.beatmap_tokenizer.get_vocab()
    assert loaded_processor.metadata_tokenizer.min_year == processor.metadata_tokenizer.min_year
