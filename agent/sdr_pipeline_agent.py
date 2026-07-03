import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from notion_client import Client

load_dotenv(override=True)


class WebResearchTool:
    """A lightweight research tool that can be extended to real web search APIs later."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    def research(self, company: str, notes: str) -> str:
        if not company:
            return "No company context available."

        if self.api_key:
            return f"External research available for {company}; using the configured search provider."

        lowered = (company + " " + notes).lower()
        if any(token in lowered for token in ["hiring", "grow", "growth", "launch", "funding", "new product", "ops"]):
            return f"Context signal detected for {company}: hiring or expansion language is present in the CRM notes."
        return f"No external research signal was available for {company}; using CRM notes as the context source."


def _get_text(prop: Optional[Dict[str, Any]]) -> str:
    if not prop:
        return ""
    prop_type = prop.get("type")
    if prop_type == "title":
        title = prop.get("title") or []
        return "".join(item.get("plain_text", "") for item in title)
    if prop_type == "rich_text":
        chunks = prop.get("rich_text") or []
        return "".join(item.get("plain_text", "") for item in chunks)
    if prop_type == "select":
        return (prop.get("select") or {}).get("name", "")
    if prop_type == "date":
        return (prop.get("date") or {}).get("start", "")
    if prop_type == "url":
        return prop.get("url", "")
    return ""


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _days_since(value: Optional[str]) -> int:
    parsed = _parse_date(value)
    if not parsed:
        return 999
    return (date.today() - parsed).days


def _activity_summary(activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not activities:
        return {
            "activity_count": 0,
            "last_activity_type": "None",
            "last_activity_date": None,
            "positive_reply": False,
            "meeting_scheduled": False,
        }

    ordered = sorted(activities, key=lambda item: item.get("date", ""), reverse=True)
    latest = ordered[0]
    positive_reply = any(
        (item.get("activity_type") or "").lower() in {"reply received", "discovery call", "meeting scheduled"}
        or (item.get("outcome") or "").lower() in {"positive reply", "completed", "meeting booked"}
        for item in ordered[:3]
    )
    meeting_scheduled = any(
        (item.get("activity_type") or "").lower() == "meeting scheduled" for item in ordered[:3]
    )
    return {
        "activity_count": len(activities),
        "last_activity_type": latest.get("activity_type") or "None",
        "last_activity_date": latest.get("date"),
        "positive_reply": positive_reply,
        "meeting_scheduled": meeting_scheduled,
    }


def analyze_relationship(record: Dict[str, Any], activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    stage = (record.get("stage") or "").lower()
    last_activity = record.get("last_activity")
    days_since = _days_since(last_activity)
    summary = _activity_summary(activities)

    if summary["positive_reply"] and days_since <= 14:
        priority = "High"
        momentum = "strong"
        reason = "The prospect has shown recent positive engagement and the relationship is warm."
    elif stage in {"call scheduled", "replied"} and days_since <= 21:
        priority = "High"
        momentum = "moderate"
        reason = "The prospect is in an active stage and likely needs a timely follow-up."
    elif stage in {"gone quiet", "contacted"} and days_since > 21:
        priority = "Medium"
        momentum = "cooling"
        reason = "The prospect has gone quiet and may need a light re-engagement."
    elif days_since > 30:
        priority = "Low"
        momentum = "inactive"
        reason = "The account has been inactive for a long time and may be better suited for nurture."
    else:
        priority = "Medium"
        momentum = "steady"
        reason = "The prospect has some recent activity but needs a careful next step."

    return {
        "company": record.get("company", ""),
        "contact_name": record.get("contact_name", ""),
        "stage": record.get("stage", ""),
        "last_activity": last_activity,
        "last_activity_days_ago": days_since,
        "momentum": momentum,
        "priority": priority,
        "reason": reason,
        "activity_summary": summary,
    }


def recommend_next_action(summary: Dict[str, Any], record: Dict[str, Any]) -> str:
    stage = (record.get("stage") or "").lower()
    days_since = summary.get("last_activity_days_ago", 999)
    positive_reply = summary.get("activity_summary", {}).get("positive_reply", False)
    meeting_scheduled = summary.get("activity_summary", {}).get("meeting_scheduled", False)

    if summary.get("priority") == "High" and positive_reply:
        return "Send a follow-up today with a personalized note and a clear next step."
    if summary.get("priority") == "High" and stage in {"call scheduled", "replied"}:
        return "Send a helpful follow-up today and share a relevant case study."
    if meeting_scheduled:
        return "Confirm the meeting and share a concise agenda or relevant proof point."
    if stage in {"gone quiet", "contacted"} and days_since > 21:
        return "Send a light nurture message this week to re-open the conversation."
    if stage in {"nurture", "closed lost"}:
        return "Keep the prospect in nurture and monitor for a better timing window."
    return "Keep the relationship warm with a short check-in and a useful insight."


def draft_follow_up_message(record: Dict[str, Any], summary: Dict[str, Any], research_context: str) -> str:
    contact_name = record.get("contact_name") or "there"
    company = record.get("company") or "the company"
    notes = record.get("notes") or ""
    reason = summary.get("reason", "")
    action = recommend_next_action(summary, record)

    base = (
        f"Hi {contact_name},\n\n"
        f"I wanted to follow up on our earlier conversation about {company}. "
        f"{reason} {research_context}\n\n"
        f"{action}\n\n"
        f"Best,\nChris"
    )

    if notes:
        base = base.replace("{company}", company)
    return base


def should_use_web_research(summary: Dict[str, Any], record: Dict[str, Any]) -> bool:
    if summary.get("priority") != "High":
        return False
    stage = (record.get("stage") or "").lower()
    return stage in {"replied", "call scheduled", "contacted"}


def get_anthropic_client() -> Optional[Anthropic]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def _extract_json_from_model_output(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    
    # Remove markdown code blocks if present
    if candidate.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", candidate, re.S)
        if match:
            candidate = match.group(1).strip()
    
    # Find JSON object or array
    if not candidate.startswith("{") and not candidate.startswith("["):
        match = re.search(r"(\{.*\}|\[.*\])", candidate, re.S)
        candidate = match.group(1) if match else candidate
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Provide helpful error info
        preview = candidate[:200] + "..." if len(candidate) > 200 else candidate
        raise ValueError(
            f"Failed to parse JSON from LLM response. Error: {e}\n"
            f"Preview: {preview}"
        ) from e


def _format_record(record: Dict[str, Any]) -> str:
    lines = []
    for key in [
        "company",
        "contact_name",
        "job_title",
        "email",
        "linkedin",
        "industry",
        "company_size",
        "hq_location",
        "lead_source",
        "owner",
        "stage",
        "last_activity",
        "next_follow_up",
        "status",
        "notes",
    ]:
        value = record.get(key, "")
        if value:
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(lines) or "No prospect details are available."


def _format_activity_timeline(activities: List[Dict[str, Any]]) -> str:
    if not activities:
        return "No activity history is available."
    ordered = sorted(activities, key=lambda item: item.get("date", ""), reverse=True)
    lines = []
    for item in ordered:
        date_text = item.get("date", "Unknown date")
        activity = item.get("activity_type", "Unknown activity")
        outcome = item.get("outcome", "")
        channel = item.get("channel", "")
        notes = item.get("notes", "")
        line = f"- {date_text}: {activity}"
        if channel:
            line += f" via {channel}"
        if outcome:
            line += f" ({outcome})"
        if notes:
            line += f" — {notes}"
        lines.append(line)
    return "\n".join(lines)


def _build_llm_prompt(record: Dict[str, Any], activities: List[Dict[str, Any]], research_context: str = "") -> str:
    record_text = _format_record(record)
    activity_text = _format_activity_timeline(activities)
    research_text = research_context or "No external research was performed."
    return (
        "You are an SDR pipeline assistant. Review the prospect record and recent activity timeline, "
        "then produce a recommended priority, the next best action, a concise reason, and a personalized draft follow-up. "
        "If external research would change your recommendation, set research_needed to \"yes\"; otherwise set it to \"no\". "
        "Return valid JSON only with these keys: research_needed, priority, next_best_action, reason, draft_follow_up, thoughts. "
        "Use priority values High, Medium, or Low. Do not add any extra keys.\n\n"
        f"Prospect record:\n{record_text}\n\n"
        f"Activity timeline:\n{activity_text}\n\n"
        f"Research context:\n{research_text}\n"
        "ADDITIONAL INSTRUCTION: The field `draft_follow_up` must be a direct, first-person message to the contact. "
        "Start the message with 'Hi {contact_name},' (replace {contact_name} with the contact's name). "
        "Do NOT include internal analysis, reasoning, or refer to the prospect in third person. "
        "Do NOT repeat the 'reason' field inside the message; keep the draft concise and actionable.\n"
    )


def llm_plan_prospect(record: Dict[str, Any], activities: List[Dict[str, Any]], research_context: str = "") -> Dict[str, Any]:
    anthropic_client = get_anthropic_client()
    if anthropic_client is not None:
        return _call_claude_planner(anthropic_client, record, activities, research_context)
    return {}



def _call_claude_planner(client: Anthropic, record: Dict[str, Any], activities: List[Dict[str, Any]], research_context: str = "") -> Dict[str, Any]:
    prompt = _build_llm_prompt(record, activities, research_context)

    model = os.getenv("ANTHROPIC_MODEL") or "claude-3.5"
    try:
        message = client.messages.create(
            model=model,
            max_tokens=900,
            system="You produce structured JSON for SDR planning.",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        hint = (
            f"Anthropic model '{model}' appears unavailable or not found for this API key. "
            "Verify your account's available models or set ANTHROPIC_MODEL to a model your account supports. "
            "You can list models with: curl https://api.anthropic.com/v1/models -H \"x-api-key: $ANTHROPIC_API_KEY\" -H \"anthropic-version: 2023-06-01\""
        )
        raise RuntimeError(f"Anthropic request failed: {e}\n{hint}") from e

    # Extract text from Anthropic SDK response (skip ThinkingBlock if present).
    text = ""
    for block in message.content:
        if hasattr(block, "text"):
            text = block.text
            break
    if not text:
        raise ValueError("No text content found in Anthropic response.")
    return _extract_json_from_model_output(text)


def _build_batch_llm_prompt(records: List[Dict[str, Any]], activities_index: Dict[str, List[Dict[str, Any]]]) -> str:
    """Build a prompt for batch planning multiple prospects in a single LLM call."""
    prospect_blocks = []
    for i, record in enumerate(records, 1):
        company_key = (record.get("company") or "").strip().lower()
        activities = activities_index.get(company_key, [])
        record_text = _format_record(record)
        activity_text = _format_activity_timeline(activities)
        prospect_blocks.append(f"## Prospect {i}: {record.get('company', 'Unknown')}\n{record_text}\n\nActivity:\n{activity_text}")
    
    prospects_section = "\n\n".join(prospect_blocks)
    return (
        "You are an SDR pipeline assistant. Review the following prospects and their activity timelines. "
        "Rank them by priority and suggest the next best action for each. "
        "The next_best_action must be specific to the prospect's company, stage, and activity context; do not reuse the same phrasing across prospects unless the evidence truly supports it.\n\n"
        "IMPORTANT: Return ONLY a valid JSON object. Do not include markdown, code blocks, or any other text.\n"
        "Return a JSON object with key 'prospects' containing an array of prospect objects.\n"
        "Each prospect object must have exactly these keys: company, contact_name, priority, next_best_action, reason, draft_follow_up, thoughts.\n"
        "Use priority values High, Medium, or Low. Rank High priorities first.\n"
        "Example format (return ONLY valid JSON, no markdown):\n"
        '{"prospects": [{"company": "...", "contact_name": "...", "priority": "High", "next_best_action": "...", "reason": "...", "draft_follow_up": "...", "thoughts": "..."}]}\n\n'
        f"{prospects_section}\n"
        "ADDITIONAL INSTRUCTION: For each prospect include a `draft_follow_up` that is a direct first-person message. "
        "Each draft must begin with 'Hi {contact_name},' and must NOT include internal analysis or refer to the prospect in third person. "
        "Keep drafts short (2-4 sentences) and actionable; do not repeat the 'reason' field verbatim.\n"
    )


def _call_claude_batch_planner(client: Anthropic, records: List[Dict[str, Any]], activities_index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Call Claude to plan multiple prospects in one batch."""
    prompt = _build_batch_llm_prompt(records, activities_index)
    model = os.getenv("ANTHROPIC_MODEL") or "claude-3.5"
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=2000,
            system="You produce structured JSON for SDR planning. Always return valid JSON with no markdown formatting.",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        hint = (
            f"Anthropic model '{model}' appears unavailable or not found for this API key. "
            "Verify your account's available models or set ANTHROPIC_MODEL to a model your account supports."
        )
        raise RuntimeError(f"Anthropic batch request failed: {e}\n{hint}") from e

    # Extract text from Anthropic SDK response, preserving every text block we get.
    text_blocks: List[str] = []
    block_types: List[str] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type:
            block_types.append(str(block_type))

        block_text = getattr(block, "text", None)
        if block_text is None and isinstance(block, dict):
            block_text = block.get("text")
        if isinstance(block_text, str) and block_text.strip():
            text_blocks.append(block_text)

    text = "\n".join(text_blocks).strip()
    if not text:
        raise ValueError(f"No text content found in Anthropic response. Block types: {block_types or ['<none>']}")
    
    response_data = _extract_json_from_model_output(text)
    return response_data.get("prospects", [])


def _deterministic_batch_plans(records: List[Dict[str, Any]], activities_index: Dict[str, List[Dict[str, Any]]], note: str) -> List[Dict[str, Any]]:
    results = []
    for record in records:
        company_key = (record.get("company") or "").strip().lower()
        activities = activities_index.get(company_key, [])
        summary = analyze_relationship(record, activities)
        next_action = recommend_next_action(summary, record)
        results.append({
            "company": record.get("company"),
            "contact_name": record.get("contact_name"),
            "priority": summary["priority"],
            "next_best_action": next_action,
            "reason": summary["reason"],
            "draft_follow_up": draft_follow_up_message(record, summary, ""),
            "thoughts": note,
        })
    return results


def _fill_missing_plan_fields(plan: Dict[str, Any], record: Dict[str, Any], activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = analyze_relationship(record, activities)
    plan.setdefault("company", record.get("company") or "Unknown")
    plan.setdefault("contact_name", record.get("contact_name") or "")
    plan.setdefault("priority", summary["priority"])
    plan.setdefault("next_best_action", recommend_next_action(summary, record))
    plan.setdefault("reason", summary["reason"])
    plan.setdefault("draft_follow_up", draft_follow_up_message(record, summary, ""))
    plan.setdefault("thoughts", "")
    return plan


def _deterministic_single_plan(record: Dict[str, Any], activities: List[Dict[str, Any]], note: str) -> Dict[str, Any]:
    summary = analyze_relationship(record, activities)
    return {
        "company": record.get("company"),
        "contact_name": record.get("contact_name"),
        "priority": summary["priority"],
        "next_best_action": recommend_next_action(summary, record),
        "reason": summary["reason"],
        "draft_follow_up": draft_follow_up_message(record, summary, ""),
        "research_context": "",
        "thoughts": note,
    }


def plan_batch_prospects(records: List[Dict[str, Any]], activities_index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Plan multiple prospects by running the single-prospect planner for each record."""
    results = []
    for record in records:
        company_key = (record.get("company") or "").strip().lower()
        activities = activities_index.get(company_key, [])
        results.append(plan_prospect(record, activities))
    return results


def plan_prospect(record: Dict[str, Any], activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Plan a single prospect (for interactive use; batch planning is preferred for efficiency)."""
    anthropic_client = get_anthropic_client()

    if anthropic_client is None:
        provider_note = "ANTHROPIC_API_KEY is not configured; using deterministic fallback reasoning."
        return _deterministic_single_plan(record, activities, provider_note)

    try:
        plan = llm_plan_prospect(record, activities)
    except Exception as exc:
        provider_note = f"Claude planning failed; using deterministic fallback reasoning. {exc}"
        return _deterministic_single_plan(record, activities, provider_note)

    if not plan:
        provider_note = "Claude returned no usable plan; using deterministic fallback reasoning."
        return _deterministic_single_plan(record, activities, provider_note)

    research_needed = (plan.get("research_needed") or "no").strip().lower()
    research_context = ""
    if research_needed == "yes":
        research_tool = WebResearchTool(api_key=os.getenv("SERPER_API_KEY"))
        research_context = research_tool.research(record.get("company", ""), record.get("notes", ""))
        try:
            plan = llm_plan_prospect(record, activities, research_context)
        except Exception as exc:
            provider_note = f"Claude re-planning after research failed; using deterministic fallback reasoning. {exc}"
            return _deterministic_single_plan(record, activities, provider_note)

        if not plan:
            provider_note = "Claude returned no usable plan after research; using deterministic fallback reasoning."
            return _deterministic_single_plan(record, activities, provider_note)

    plan["research_context"] = research_context
    plan["company"] = record.get("company")
    plan["contact_name"] = record.get("contact_name")
    if "thoughts" not in plan:
        plan["thoughts"] = "No explicit thoughts returned."
    return plan


def get_notion_client() -> Client:
    token = os.getenv("NOTION_TOKEN")
    if not token:
        raise RuntimeError("NOTION_TOKEN is missing in the environment.")
    return Client(auth=token)


def query_all_pages(notion: Client, data_source_id: str) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    start_cursor: Optional[str] = None

    while True:
        kwargs = {"data_source_id": data_source_id}
        if start_cursor:
            kwargs["start_cursor"] = start_cursor

        response = notion.data_sources.query(**kwargs)
        all_results.extend(response.get("results", []))

        if not response.get("has_more"):
            break
        start_cursor = response.get("next_cursor")

    return all_results


def prospect_to_record(page: Dict[str, Any]) -> Dict[str, Any]:
    properties = page.get("properties", {})
    return {
        "page_id": page.get("id"),
        "company": _get_text(properties.get("Company")),
        "contact_name": _get_text(properties.get("Contact Name")),
        "job_title": _get_text(properties.get("Job Title")),
        "email": _get_text(properties.get("Email")),
        "linkedin": _get_text(properties.get("LinkedIn")),
        "industry": _get_text(properties.get("Industry")),
        "company_size": _get_text(properties.get("Company Size")),
        "hq_location": _get_text(properties.get("HQ Location")),
        "lead_source": _get_text(properties.get("Lead Source")),
        "owner": _get_text(properties.get("Owner")),
        "stage": _get_text(properties.get("Stage")),
        "last_activity": _get_text(properties.get("Last Activity")),
        "next_follow_up": _get_text(properties.get("Next Follow-up")),
        "status": _get_text(properties.get("Status")),
        "notes": _get_text(properties.get("Notes")),
        "priority": _get_text(properties.get("Priority")),
        "next_best_action": _get_text(properties.get("Next Best Action")),
        "reason": _get_text(properties.get("Reason")),
        "draft_follow_up": _get_text(properties.get("Draft Follow-up")),
    }


def activity_to_record(page: Dict[str, Any]) -> Dict[str, Any]:
    properties = page.get("properties", {})
    return {
        "date": _get_text(properties.get("Date")),
        "company": _get_text(properties.get("Name")),
        "contact_name": _get_text(properties.get("Contact Name")),
        "activity_type": _get_text(properties.get("Activity Type")),
        "channel": _get_text(properties.get("Channel")),
        "outcome": _get_text(properties.get("Outcome")),
        "notes": _get_text(properties.get("Notes")),
    }


def build_activity_index(activity_pages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    activity_index: Dict[str, List[Dict[str, Any]]] = {}
    for page in activity_pages:
        activity = activity_to_record(page)
        company = (activity.get("company") or "").strip()
        if not company:
            continue
        activity_index.setdefault(company.lower(), []).append(activity)
    return activity_index


def update_prospect_recommendation(notion: Client, page_id: str, recommendation: Dict[str, Any]) -> None:
    priority = recommendation["priority"]
    next_action = recommendation["next_best_action"]
    reason = recommendation["reason"]
    draft_message = recommendation["draft_follow_up"]

    next_follow_up = date.today()
    if "follow-up" in next_action.lower() or "follow up" in next_action.lower():
        next_follow_up = date.today() + timedelta(days=2)
    elif "nurture" in next_action.lower():
        next_follow_up = date.today() + timedelta(days=7)
    elif "confirm" in next_action.lower() or "agenda" in next_action.lower():
        next_follow_up = date.today() + timedelta(days=1)

    properties = {
        "Priority": {"rich_text": [{"type": "text", "text": {"content": priority}}]},
        "Next Best Action": {"rich_text": [{"type": "text", "text": {"content": next_action}}]},
        "Reason": {"rich_text": [{"type": "text", "text": {"content": reason}}]},
        "Draft Follow-up": {"rich_text": [{"type": "text", "text": {"content": draft_message}}]},
        "Next Follow-up": {"date": {"start": next_follow_up.isoformat()}},
    }
    notion.pages.update(page_id=page_id, properties=properties)


def create_recommendation_entry(notion: Client, prospect_page_id: str, recommendation: Dict[str, Any]) -> None:
    """Create a new page in the Recommendations database with the given recommendation.

    Requires environment variable `RECOMMENDATIONS_DB_ID` to be set to the target database id.
    """
    db_id = os.getenv("RECOMMENDATIONS_DB_ID")
    if not db_id:
        raise RuntimeError("RECOMMENDATIONS_DB_ID is missing in the environment. Set it to the target Notion database ID.")

    company = recommendation.get("company", "")
    contact = recommendation.get("contact_name", "")
    priority = recommendation.get("priority", "")
    next_action = recommendation.get("next_best_action", "")
    reason = recommendation.get("reason", "")
    draft = recommendation.get("draft_follow_up", "")
    thoughts = recommendation.get("thoughts", "")

    prospect_link_value: Dict[str, Any] = {"url": f"https://www.notion.so/{prospect_page_id.replace('-', '')}"}
    try:
        schema = notion.databases.retrieve(database_id=db_id)
        prospect_prop = (schema.get("properties") or {}).get("Prospect Page") or {}
        prop_type = prospect_prop.get("type")
        if prop_type == "relation":
            prospect_link_value = {"relation": [{"id": prospect_page_id}]}
        elif prop_type == "rich_text":
            prospect_link_value = {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "Open prospect",
                            "link": {"url": f"https://www.notion.so/{prospect_page_id.replace('-', '')}"},
                        },
                    }
                ]
            }
    except Exception:
        # If schema discovery fails, keep URL fallback so writes are not blocked.
        pass

    properties: Dict[str, Any] = {
        "Company": {"title": [{"type": "text", "text": {"content": company}}]},
        "Contact Name": {"rich_text": [{"type": "text", "text": {"content": contact}}]},
        "Priority": {"select": {"name": priority}},
        "Date": {"date": {"start": date.today().isoformat()}},
        "Next Best Action": {"rich_text": [{"type": "text", "text": {"content": next_action}}]},
        "Reason": {"rich_text": [{"type": "text", "text": {"content": reason}}]},
        "Draft Follow-up": {"rich_text": [{"type": "text", "text": {"content": draft}}]},
        "Thoughts": {"rich_text": [{"type": "text", "text": {"content": thoughts}}]},
        "Prospect Page": prospect_link_value,
    }

    # If the recommendations database has a people-based Contact Name property, ignore it to avoid invalid payloads.
    notion.pages.create(parent={"database_id": db_id}, properties=properties)


def _priority_sort_key(recommendation: Dict[str, Any]) -> int:
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    return priority_order.get(recommendation.get("priority", "Medium"), 1)


def build_daily_review_snapshot(limit: int = 10) -> List[Dict[str, Any]]:
    notion = get_notion_client()
    prospects_db_id = os.getenv("PROSPECTS_DB_ID")
    activity_db_id = os.getenv("ACTIVITY_DB_ID")

    if not prospects_db_id or not activity_db_id:
        raise RuntimeError("PROSPECTS_DB_ID or ACTIVITY_DB_ID is missing in the environment.")

    prospect_pages = query_all_pages(notion, prospects_db_id)
    activity_pages = query_all_pages(notion, activity_db_id)
    activity_index = build_activity_index(activity_pages)

    # Batch plan ALL prospects (not just first N) so Claude can rank globally
    records = [prospect_to_record(p) for p in prospect_pages]
    recommendations = plan_batch_prospects(records, activity_index)

    # Sort by priority (High > Medium > Low) to get the best prospects first
    recommendations.sort(key=_priority_sort_key)

    # Take only the top `limit` recommendations
    top_recommendations = recommendations[:limit]
    records_by_company = {
        (record.get("company") or "").strip().lower(): record
        for record in records
        if (record.get("company") or "").strip()
    }

    for recommendation in top_recommendations:
        matching_record = records_by_company.get((recommendation.get("company") or "").strip().lower())
        if matching_record and matching_record.get("page_id"):
            recommendation["page_id"] = matching_record["page_id"]

    return top_recommendations


def publish_recommendations(notion: Client, recommendations: List[Dict[str, Any]]) -> None:
    """Persist a precomputed recommendation snapshot to the Recommendations database."""
    db_id = os.getenv("RECOMMENDATIONS_DB_ID")
    if not db_id:
        raise RuntimeError(
            "RECOMMENDATIONS_DB_ID is not set. To avoid overwriting prospect fields, set RECOMMENDATIONS_DB_ID in your .env to a Recommendations DB ID before publishing recommendations."
        )

    for recommendation in recommendations:
        prospect_page_id = recommendation.get("page_id")
        if not prospect_page_id:
            raise RuntimeError(
                f"Recommendation for {recommendation.get('company', 'Unknown')} is missing page_id. Build the snapshot before publishing so the same top N records are written to Notion."
            )
        create_recommendation_entry(notion, prospect_page_id, recommendation)


def run_daily_review(limit: int = 10, dry_run: bool = False) -> List[Dict[str, Any]]:
    top_recommendations = build_daily_review_snapshot(limit)

    if not dry_run:
        notion = get_notion_client()
        publish_recommendations(notion, top_recommendations)

    return top_recommendations
