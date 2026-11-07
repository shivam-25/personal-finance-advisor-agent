import os
from pathlib import Path

import pytest

# Isolate tests from any local .env: no network, no shared memory store.
os.environ["ADVISOR_DISABLE_LLM"] = "1"
os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)

from advisor.bank import BankAPI
from advisor.llm import LLMClient
from advisor.memory import JsonFileBackend, MemoryStore
from advisor.orchestrator import AdvisorRuntime
from advisor.tools import ToolContext, ToolRegistry

CUSTOMER = "CUST001"


@pytest.fixture
def bank():
    return BankAPI()


@pytest.fixture
def backend(tmp_path: Path):
    return JsonFileBackend(tmp_path / "memory")


@pytest.fixture
def memory(backend):
    return MemoryStore(CUSTOMER, "test", backend=backend)


@pytest.fixture
def registry(bank, memory):
    return ToolRegistry(ToolContext(bank, CUSTOMER, memory))


@pytest.fixture
def llm():
    return LLMClient()


@pytest.fixture
def runtime(backend):
    return AdvisorRuntime(backend=backend)


@pytest.fixture
def session(runtime):
    return runtime.session(CUSTOMER, "test")


@pytest.fixture
def transactions(bank):
    from advisor.categorizer import categorize_all

    return categorize_all(bank.get_transactions(CUSTOMER))
