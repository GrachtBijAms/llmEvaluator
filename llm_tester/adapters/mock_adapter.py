from llm_tester.adapters.base import BaseLLMAdapter

class MockLLMAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return f"MOCK_RESPONSE: {prompt}"
