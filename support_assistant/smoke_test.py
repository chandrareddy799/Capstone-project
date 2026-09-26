"""Exercise the policy and general-question paths without starting a server."""
import json

from main import AskRequest, ask


examples = {
    "policy question": "How long does delivery take?",
    "general question": "What is the weather today?",
}

for label, question in examples.items():
    response = ask(AskRequest(query=question))
    print(f"{label}: {question}")
    print(json.dumps(response.model_dump(), ensure_ascii=True))
