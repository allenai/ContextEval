"""

Main classes for ContextEval

@kylel

"""

from typing import Dict, List


class DummyContextEval:
    def __init__(self, *args, **kwargs):
        pass

    def generate_contexts(self, query: str, answer: str, *args, **kwargs):
        return [{"q": "What?", "a": "This."}, {"q": "Why?", "a": "Because."}]

    def evaluate(self, contexts: List[Dict[str, str]], *args, **kwargs):
        return {"score": 0.5, "contexts": contexts}
