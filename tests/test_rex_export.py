import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rex_export import load_export, prompt_reply_pairs  # noqa: E402

METADATA = {
    "source": "test",
    "generator-id": "rex-history-visit",
    "timestamp": 1790181327.638,
    "timezone": "Europe/Berlin",
}


def visit(visit_id, url, domain, visit_time_ms, transition="link"):
    return {
        "name": "rex-history-visit",
        "url": url,
        "recorded_url": url,
        "domain": domain,
        "title": url,
        "visit_time": visit_time_ms,
        "transition_type": transition,
        "visit_id": visit_id,
        "referring_visit_id": "0",
        "visit_count": 1,
        "typed_count": 0,
        "passive-data-metadata": METADATA,
    }


def message(node_id, role, content, create_time, recipient="all", metadata=None, author_name=None):
    return {
        "id": node_id,
        "author": {"role": role, "name": author_name, "metadata": {}},
        "create_time": create_time,
        "update_time": create_time + 1,
        "content": content,
        "status": "finished_successfully",
        "recipient": recipient,
        "metadata": metadata or {},
    }


def memory_write_conversation():
    """A user message, a memory write, its tool result, and the visible reply.

    `turns` is left empty on purpose: the loader must read messages from
    `metadata.mapping`, which is the complete tree.
    """
    mapping = {
        "client-created-root": {"id": "client-created-root", "message": None, "parent": None, "children": ["u1"]},
        "u1": {"id": "u1", "parent": "client-created-root", "children": ["a1"], "message": message(
            "u1", "user", {"content_type": "text", "parts": ["my tortoise is thirty two"]}, 1790262170.0)},
        "a1": {"id": "a1", "parent": "u1", "children": ["t1"], "message": message(
            "a1", "assistant", {"content_type": "code", "language": "unknown", "text": "The user's tortoise is 32."},
            1790262175.0, recipient="bio")},
        "t1": {"id": "t1", "parent": "a1", "children": ["a2"], "message": message(
            "t1", "tool", {"content_type": "text", "parts": ["Model set context updated."]},
            1790262176.0, recipient="assistant", author_name="bio")},
        "a2": {"id": "a2", "parent": "t1", "children": [], "message": message(
            "a2", "assistant", {"content_type": "text", "parts": ["That means 30 years together."]},
            1790262178.0, metadata={
                "model_slug": "gpt-5-6",
                "resolved_model_slug": "gpt-5-6",
                "default_model_slug": "auto",
                "finished_duration_sec": 4,
            })},
    }
    return {
        "name": "rex-conversation",
        "platform": "chatgpt",
        "identifier": "conv-1",
        "turns": [],
        "metadata": {
            "title": "Tortoise Age Calculation",
            "create_time": 1790262170.0,
            "update_time": 1790262183.5,
            "conversation_id": "conv-1",
            "default_model_slug": "auto",
            "is_temporary_chat": False,
            "mapping": mapping,
        },
        "passive-data-metadata": METADATA,
    }


@pytest.fixture
def export(tmp_path):
    records = [
        visit("1", "DOMAIN ONLY", "google.com", 1790181327638.0, transition="form_submit"),
        visit("2", "CATEGORY:email", "", 1790181400000.0),
        visit("3", "https://github.com/bric-digital", "github.com", 1790181500000.0, transition="typed"),
        memory_write_conversation(),
        {"name": "pdk-app-event", "event_name": "rex-spider-chatgpt-complete",
         "event_details": {"crawled_count": 1, "date": 1790784475712}, "passive-data-metadata": METADATA},
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(records))
    return load_export(path)


def test_visits_are_classified_by_privacy_tier(export):
    tiers = export.visits.set_index("visit_id")["privacy_tier"].to_dict()
    assert tiers == {"1": "domain", "2": "category", "3": "full"}


def test_category_label_is_taken_from_the_url(export):
    categories = export.visits.set_index("visit_id")["category"].to_dict()
    assert categories["2"] == "email"
    assert pd.isna(categories["1"]) and pd.isna(categories["3"])


def test_visit_time_is_utc(export):
    first = export.visits.set_index("visit_id").loc["1", "visit_time"]
    assert first == pd.Timestamp("2026-09-23T16:35:27.638", tz="UTC")


def test_messages_come_from_the_mapping_not_the_turns(export):
    assert export.messages["node_id"].tolist() == ["u1", "a1", "t1", "a2"]


def test_hidden_steps_are_the_memory_write_and_its_result(export):
    hidden = export.messages.set_index("node_id")["is_hidden_step"].to_dict()
    assert hidden == {"u1": False, "a1": True, "t1": True, "a2": False}


def test_tool_name_is_the_call_recipient_or_the_result_author(export):
    tool_names = export.messages.set_index("node_id")["tool_name"].to_dict()
    assert tool_names["a1"] == "bio" and tool_names["t1"] == "bio"
    assert pd.isna(tool_names["u1"]) and pd.isna(tool_names["a2"])


def test_message_text_covers_parts_and_code(export):
    text = export.messages.set_index("node_id")["text"].to_dict()
    assert text["a1"] == "The user's tortoise is 32."
    assert text["a2"] == "That means 30 years together."


def test_reply_carries_requested_and_resolved_model(export):
    reply = export.messages.set_index("node_id").loc["a2"]
    assert (reply["default_model_slug"], reply["resolved_model_slug"], reply["finished_duration_sec"]) == ("auto", "gpt-5-6", 4)


def test_conversation_row(export):
    conv = export.conversations.set_index("conversation_id").loc["conv-1"]
    assert conv["title"] == "Tortoise Age Calculation"
    assert conv["create_time"] == pd.Timestamp(1790262170.0, unit="s", tz="UTC")


def test_capture_timezone(export):
    assert export.capture_timezone == "Europe/Berlin"


def test_events(export):
    assert export.events["event_name"].tolist() == ["rex-spider-chatgpt-complete"]


def test_reply_pairs_with_its_prompt_across_hidden_steps(export):
    pairs = prompt_reply_pairs(export.messages)
    assert pairs[["prompt_id", "reply_id", "prompt_chars", "reply_chars"]].values.tolist() == [
        ["u1", "a2", len("my tortoise is thirty two"), len("That means 30 years together.")]]
