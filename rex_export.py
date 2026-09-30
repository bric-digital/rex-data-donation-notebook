"""Load a REX local-download export into pandas tables.

    from rex_export import load_export
    export = load_export("rex-export.json")
    export.visits, export.conversations, export.messages, export.events

The export is one JSON array mixing three record types: `rex-history-visit`,
`rex-conversation` and `pdk-app-event`. Messages are read from each
conversation's `metadata.mapping`, which is ChatGPT's complete message tree.
`turns` holds a copy of the same nodes and is not read.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DOMAIN_ONLY = "DOMAIN ONLY"
CATEGORY_PREFIX = "CATEGORY:"

VISIBLE_RECIPIENT = "all"


@dataclass
class RexExport:
    visits: pd.DataFrame
    conversations: pd.DataFrame
    messages: pd.DataFrame
    events: pd.DataFrame
    capture_timezone: str | None


def load_export(path) -> RexExport:
    records = json.loads(Path(path).read_text())
    by_name = {}
    for record in records:
        by_name.setdefault(record.get("name"), []).append(record)

    conversation_records = by_name.get("rex-conversation", [])
    return RexExport(
        visits=visits_table(by_name.get("rex-history-visit", [])),
        conversations=conversations_table(conversation_records),
        messages=messages_table(conversation_records),
        events=events_table(by_name.get("pdk-app-event", [])),
        capture_timezone=capture_timezone(records),
    )


def privacy_tier(url: str) -> str:
    if url == DOMAIN_ONLY:
        return "domain"
    if url.startswith(CATEGORY_PREFIX):
        return "category"
    return "full"


def category_label(url: str) -> str | None:
    if url.startswith(CATEGORY_PREFIX):
        return url[len(CATEGORY_PREFIX):]
    return None


def is_hidden_step(message: dict) -> bool:
    """A step the model took that the user never sees as a reply.

    Tool results, and assistant messages addressed to a tool (a memory write
    to `bio`, a connector call) rather than to the user.
    """
    if message["author"]["role"] == "tool":
        return True
    return message.get("recipient", VISIBLE_RECIPIENT) != VISIBLE_RECIPIENT


def tool_name(message: dict) -> str | None:
    """The tool a hidden step talks to: the recipient of a call, or the author of a result."""
    if not is_hidden_step(message):
        return None
    if message["author"]["role"] == "tool":
        return message["author"].get("name")
    return message.get("recipient")


def message_text(content: dict) -> str:
    if "text" in content:
        return content["text"]
    if "content" in content and isinstance(content["content"], str):
        return content["content"]
    parts = content.get("parts") or []
    return "\n".join(part for part in parts if isinstance(part, str))


def capture_timezone(records: list) -> str | None:
    for record in records:
        timezone = (record.get("passive-data-metadata") or {}).get("timezone")
        if timezone:
            return timezone
    return None


def epoch_seconds(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values, dtype="float64"), unit="s", utc=True)


def visits_table(records: list) -> pd.DataFrame:
    rows = [{
        "visit_id": r["visit_id"],
        "referring_visit_id": r.get("referring_visit_id"),
        "visit_time_ms": r["visit_time"],
        "domain": r.get("domain") or "",
        "transition_type": r.get("transition_type"),
        "privacy_tier": privacy_tier(r["url"]),
        "category": category_label(r["url"]),
    } for r in records]
    table = pd.DataFrame(rows, columns=[
        "visit_id", "referring_visit_id", "visit_time_ms", "domain",
        "transition_type", "privacy_tier", "category"])
    table.insert(2, "visit_time", pd.to_datetime(table.pop("visit_time_ms"), unit="ms", utc=True))
    return table.sort_values("visit_time", ignore_index=True)


def conversations_table(records: list) -> pd.DataFrame:
    rows = [{
        "conversation_id": r["identifier"],
        "title": r["metadata"].get("title"),
        "create_time": r["metadata"].get("create_time"),
        "update_time": r["metadata"].get("update_time"),
        "default_model_slug": r["metadata"].get("default_model_slug"),
        "is_temporary_chat": r["metadata"].get("is_temporary_chat"),
    } for r in records]
    table = pd.DataFrame(rows, columns=[
        "conversation_id", "title", "create_time", "update_time",
        "default_model_slug", "is_temporary_chat"])
    table["create_time"] = epoch_seconds(table["create_time"])
    table["update_time"] = epoch_seconds(table["update_time"])
    return table.sort_values("create_time", ignore_index=True)


def messages_table(records: list) -> pd.DataFrame:
    rows = []
    for record in records:
        for node_id, node in record["metadata"]["mapping"].items():
            message = node.get("message")
            if message is None:
                continue
            metadata = message.get("metadata") or {}
            rows.append({
                "conversation_id": record["identifier"],
                "node_id": node_id,
                "parent_id": node.get("parent"),
                "role": message["author"]["role"],
                "recipient": message.get("recipient"),
                "content_type": message["content"].get("content_type"),
                "text": message_text(message["content"]),
                "create_time": message.get("create_time"),
                "status": message.get("status"),
                "model_slug": metadata.get("model_slug"),
                "resolved_model_slug": metadata.get("resolved_model_slug"),
                "default_model_slug": metadata.get("default_model_slug"),
                "finished_duration_sec": metadata.get("finished_duration_sec"),
                "is_hidden_step": is_hidden_step(message),
                "tool_name": tool_name(message),
            })
    table = pd.DataFrame(rows, columns=[
        "conversation_id", "node_id", "parent_id", "role", "recipient",
        "content_type", "text", "create_time", "status", "model_slug",
        "resolved_model_slug", "default_model_slug", "finished_duration_sec",
        "is_hidden_step", "tool_name"])
    table["create_time"] = epoch_seconds(table["create_time"])
    return table.sort_values(["conversation_id", "create_time"], ignore_index=True)


def events_table(records: list) -> pd.DataFrame:
    rows = [{
        "event_name": r.get("event_name"),
        "date": (r.get("event_details") or {}).get("date"),
        "details": r.get("event_details"),
    } for r in records]
    table = pd.DataFrame(rows, columns=["event_name", "date", "details"])
    table["date"] = pd.to_datetime(table["date"], unit="ms", utc=True)
    return table


def prompt_reply_pairs(messages: pd.DataFrame) -> pd.DataFrame:
    """Each visible assistant reply paired with the user prompt it answers.

    The prompt is the nearest user message up the tree, so hidden steps
    between the two (a memory write, a tool call) are skipped over.
    """
    by_id = messages.set_index("node_id")
    replies = messages[(messages["role"] == "assistant") & ~messages["is_hidden_step"]
                       & (messages["content_type"] == "text")]
    rows = []
    for reply in replies.itertuples():
        parent = reply.parent_id
        while parent in by_id.index and by_id.loc[parent, "role"] != "user":
            parent = by_id.loc[parent, "parent_id"]
        if parent in by_id.index:
            rows.append({
                "conversation_id": reply.conversation_id,
                "prompt_id": parent,
                "reply_id": reply.node_id,
                "prompt_chars": len(by_id.loc[parent, "text"]),
                "reply_chars": len(reply.text),
            })
    return pd.DataFrame(rows, columns=[
        "conversation_id", "prompt_id", "reply_id", "prompt_chars", "reply_chars"])
