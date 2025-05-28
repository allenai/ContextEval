"""

Tests for main ContextEval

@kylel

"""

import unittest

from contexteval.contexteval import DummyContextEval


class TestContextEval(unittest.TestCase):
    def test_generate_contexts(self):
        ce = DummyContextEval()
        contexts = ce.generate_contexts("query", "answer")
        self.assertEqual(contexts, [{"q": "What?", "a": "This."}, {"q": "Why?", "a": "Because."}])

    def test_evaluate(self):
        ce = DummyContextEval()
        contexts = [{"q": "What?", "a": "This."}, {"q": "Why?", "a": "Because."}]
        result = ce.evaluate(contexts)
        self.assertEqual(result, {"score": 0.5, "contexts": contexts})
