.PHONY: install test serve demo chat agents data

install:
	python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

test:
	./.venv/bin/python -m pytest

serve:
	./.venv/bin/python -m uvicorn advisor.server:app --reload --port 8000

demo:
	./.venv/bin/python -m advisor.cli demo

chat:
	./.venv/bin/python -m advisor.cli chat

agents:
	./.venv/bin/python -m advisor.cli agents

data:
	./.venv/bin/python scripts/generate_mock_data.py
