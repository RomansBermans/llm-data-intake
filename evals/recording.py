"""Keep the exact inputs and outcomes used in conversation evaluations."""

from copy import deepcopy
import json

from src.extractor import ModelProviderError


class CachedExtractor:
    """Reuse one extraction for identical inputs across verifier configurations."""

    def __init__(self, extractor):
        self.extractor = extractor
        self.cache = {}

    def extract(self, message, state, role, context):
        key = json.dumps([message, state, role, context], sort_keys=True)
        if key not in self.cache:
            self.cache[key] = self.extractor.extract(message, state, role, context)
        return deepcopy(self.cache[key])


class RecordingVerifier:
    def __init__(self, verifier, comparison=None):
        self.verifier = verifier
        self.comparison = comparison
        self.current = None

    def verify(self, message, state, role, context, extraction):
        self.current = deepcopy({"message": message, "state": state, "role": role,
                                 "context": context, "extraction": extraction})
        try:
            decision = self.verifier.verify(message, state, role, context, extraction)
            self.current["verification"] = deepcopy(decision)
        except ModelProviderError as error:
            self.current["error"] = str(error)
            raise
        if self.comparison:
            try:
                self.current["comparison_verification"] = self.comparison.verify(
                    message, state, role, context, deepcopy(extraction))
            except ModelProviderError as error:
                self.current["comparison_error"] = str(error)
        return decision
