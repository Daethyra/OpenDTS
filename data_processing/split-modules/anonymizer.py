from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import AnonymizerConfig

class Anonymizer:
    def __init__(self):
        # Initialize the engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact_pii(self, text):
        # Analyze the text
        analyzer_results = self.analyzer.analyze(text=text, language='en')

        # Define the anonymizer configuration
        anonymizer_config = AnonymizerConfig("replace", {"new_value": "[REDACTED]"})

        # Anonymize the text
        anonymized_text = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            anonymizers_config={"DEFAULT": anonymizer_config}
        )

        return anonymized_text.text
