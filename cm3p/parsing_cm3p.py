from transformers import FeatureExtractionMixin


class CM3PBeatmapParser(FeatureExtractionMixin):
    """
    A class to parse CM3P beatmap files.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.beatmap_data = {}

    def parse(self):
        """
        Parses the CM3P beatmap file and stores the data in a dictionary.
        """
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    self.beatmap_data[key.strip()] = value.strip()
        return self.beatmap_data
