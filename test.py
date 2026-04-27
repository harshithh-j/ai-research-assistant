import httpx

with httpx.stream("POST", "http://localhost:8000/api/v1/chat", json={
    "messages": [{"role": "user", "content": "What is RAG in 3 sentences?"}],
    "stream": True
}) as r:
    for chunk in r.iter_text():
        print(chunk, end="", flush=True)
print()
