from advisor.memory import ConversationMemory, JsonFileBackend, MemoryStore, ProfileMemory


def test_conversation_memory_is_bounded():
    memory = ConversationMemory(max_turns=4)
    for i in range(10):
        memory.add("user", f"m{i}")
    assert len(memory.turns) == 4
    assert memory.turns[-1].content == "m9"


def test_last_intent_reads_the_most_recent_user_turn():
    memory = ConversationMemory()
    memory.add("user", "a", intent="spend_review")
    memory.add("assistant", "b")
    memory.add("user", "c", intent="budget_alerts")
    assert memory.last_intent() == "budget_alerts"
    assert memory.last_user_turn().content == "c"


def test_working_memory_is_cleared_each_turn(memory):
    memory.working.put("spend.analysis", {"x": 1})
    assert memory.working.has("spend.analysis")
    memory.start_turn()
    assert not memory.working.has("spend.analysis")


def test_conversation_survives_a_new_turn(memory):
    memory.conversation.add("user", "hello")
    memory.start_turn()
    assert len(memory.conversation.turns) == 1


def test_profile_memory_upserts_on_kind_and_key(memory):
    memory.profile.remember("preference", "Travel", "protected")
    memory.profile.remember("preference", "Travel", "allowed")
    notes = memory.profile.all("preference")
    assert len(notes) == 1
    assert notes[0].value == "allowed"


def test_profile_memory_persists_across_instances(tmp_path):
    backend = JsonFileBackend(tmp_path / "mem")
    first = ProfileMemory("CUST001", backend=backend)
    first.remember("preference", "Travel", "protected", "stated by customer")
    second = ProfileMemory("CUST001", backend=backend)
    assert second.constraints()["protected_categories"] == ["Travel"]
    assert second.all("preference")[0].note == "stated by customer"


def test_memory_is_scoped_per_customer(tmp_path):
    backend = JsonFileBackend(tmp_path / "mem")
    ProfileMemory("CUST001", backend=backend).remember("preference", "Travel", "protected")
    assert ProfileMemory("CUST002", backend=backend).all() == []


def test_forget_removes_a_note(memory):
    memory.profile.remember("preference", "Travel", "protected")
    assert memory.profile.forget("preference", "Travel") is True
    assert memory.profile.forget("preference", "Travel") is False
    assert memory.profile.constraints()["protected_categories"] == []


def test_recall_ranks_by_keyword_overlap(memory):
    memory.profile.remember("preference", "Travel", "protected", "holidays matter")
    memory.profile.remember("goal", "Laptop", "60000", "saving for a new laptop")
    assert memory.profile.recall("travel holidays")[0].key == "Travel"
    assert memory.profile.recall("laptop")[0].key == "Laptop"


def test_recall_ignores_stopwords(memory):
    memory.profile.remember("preference", "Travel", "protected")
    assert memory.profile.recall("what is the my a an") == memory.profile.all()[-5:]


def test_constraints_group_by_kind(memory):
    memory.profile.remember("preference", "Travel", "protected")
    memory.profile.remember("declined_recommendation", "subscriptions", "declined")
    memory.profile.remember("goal", "Trip", "250000")
    constraints = memory.profile.constraints()
    assert constraints["protected_categories"] == ["Travel"]
    assert constraints["declined"] == ["subscriptions"]
    assert constraints["goals"][0]["key"] == "Trip"


def test_snapshot_reports_the_backend(memory):
    snapshot = memory.snapshot()
    assert snapshot["profile"]["backend"].startswith("json:")
    assert "conversation" in snapshot and "working" in snapshot


def test_write_failure_does_not_raise(tmp_path):
    class Broken(JsonFileBackend):
        def save(self, customer_id, notes):
            raise OSError("read-only")

    profile = ProfileMemory("CUST001", backend=JsonFileBackend(tmp_path))
    profile.backend = Broken(tmp_path)
    profile.remember("preference", "Travel", "protected")
    assert profile.constraints()["protected_categories"] == ["Travel"]


def test_memory_changes_the_advice_across_turns(session):
    before = session.ask("How can I improve my monthly savings?")
    assert "Food & Dining" in {r["category"] for r in before.recommendations}

    session.ask("Don't suggest cutting my food spending")

    after = session.ask("How can I improve my monthly savings?")
    assert "Food & Dining" not in {r["category"] for r in after.recommendations}
    assert after.memory["profile"]["constraints"]["protected_categories"] == ["Food & Dining"]
