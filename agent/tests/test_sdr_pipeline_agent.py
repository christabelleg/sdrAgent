import os
import unittest
from unittest.mock import MagicMock

from sdr_pipeline_agent import (
    analyze_relationship,
    plan_batch_prospects,
    plan_prospect,
    publish_recommendations,
    recommend_next_action,
)


class SdrPipelineAgentTests(unittest.TestCase):
    def test_positive_reply_earns_follow_up_recommendation(self):
        record = {
            "company": "Watershed",
            "stage": "Replied",
            "last_activity": "2026-06-20",
            "notes": "Prospect replied positively and asked for more context.",
        }
        activities = [
            {"date": "2026-06-18", "activity_type": "Reply Received", "outcome": "Positive reply"},
            {"date": "2026-06-10", "activity_type": "Discovery Call", "outcome": "Completed"},
        ]

        summary = analyze_relationship(record, activities)
        recommendation = recommend_next_action(summary, record)

        self.assertEqual(summary["priority"], "High")
        self.assertIn("follow-up", recommendation.lower())

    def test_quiet_prospect_gets_nurture_recommendation(self):
        record = {
            "company": "Front",
            "stage": "Gone Quiet",
            "last_activity": "2026-06-01",
            "notes": "No response for several weeks.",
        }
        activities = [
            {"date": "2026-05-20", "activity_type": "Email Sent", "outcome": "Sent"},
            {"date": "2026-05-10", "activity_type": "No Response", "outcome": "No response"},
        ]

        summary = analyze_relationship(record, activities)
        recommendation = recommend_next_action(summary, record)

        self.assertEqual(summary["priority"], "Medium")
        self.assertIn("nurture", recommendation.lower())

    def test_plan_prospect_fallback_returns_thoughts_without_anthropic_key(self):
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            record = {
                "company": "Front",
                "stage": "Gone Quiet",
                "last_activity": "2026-06-01",
                "notes": "No response for several weeks.",
            }
            activities = [
                {"date": "2026-05-20", "activity_type": "Email Sent", "outcome": "Sent"},
                {"date": "2026-05-10", "activity_type": "No Response", "outcome": "No response"},
            ]
            recommendation = plan_prospect(record, activities)
            self.assertEqual(recommendation["priority"], "Medium")
            self.assertIn("thoughts", recommendation)
            self.assertTrue(recommendation["draft_follow_up"])
        finally:
            if old_anthropic is not None:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    def test_publish_recommendations_writes_snapshot_into_recommendations_db(self):
        old_db_id = os.environ.get("RECOMMENDATIONS_DB_ID")
        os.environ["RECOMMENDATIONS_DB_ID"] = "recommendations-db-id"
        try:
            notion = MagicMock()
            recommendations = [
                {
                    "company": "Front",
                    "contact_name": "Avery",
                    "priority": "High",
                    "next_best_action": "Send a follow-up today.",
                    "reason": "Recent positive engagement.",
                    "draft_follow_up": "Hi Avery,\n\nFollowing up on our earlier conversation...",
                    "thoughts": "Snapshot test",
                    "page_id": "page-123",
                }
            ]

            publish_recommendations(notion, recommendations)

            notion.pages.create.assert_called_once()
            kwargs = notion.pages.create.call_args.kwargs
            self.assertEqual(kwargs["parent"]["database_id"], "recommendations-db-id")
            self.assertEqual(kwargs["properties"]["Prospect Page"]["url"], "https://www.notion.so/page-123")
            self.assertEqual(kwargs["properties"]["Company"]["title"][0]["text"]["content"], "Front")
        finally:
            if old_db_id is None:
                os.environ.pop("RECOMMENDATIONS_DB_ID", None)
            else:
                os.environ["RECOMMENDATIONS_DB_ID"] = old_db_id

    def test_plan_batch_prospects_uses_each_record_individually(self):
        records = [
            {"company": "Hex", "contact_name": "A", "stage": "Replied", "last_activity": "2026-06-20"},
            {"company": "Miro", "contact_name": "B", "stage": "Gone Quiet", "last_activity": "2026-06-01"},
        ]
        activities_index = {}

        original_plan_prospect = __import__("sdr_pipeline_agent").plan_prospect
        try:
            import sdr_pipeline_agent as agent_module

            def fake_plan_prospect(record, activities):
                return {
                    "company": record["company"],
                    "contact_name": record["contact_name"],
                    "priority": "High" if record["company"] == "Hex" else "Medium",
                    "next_best_action": f"Next action for {record['company']}",
                    "reason": f"Reason for {record['company']}",
                    "draft_follow_up": f"Hi {record['contact_name']},",
                    "thoughts": "mocked",
                }

            agent_module.plan_prospect = fake_plan_prospect
            plans = plan_batch_prospects(records, activities_index)

            self.assertEqual(plans[0]["next_best_action"], "Next action for Hex")
            self.assertEqual(plans[1]["next_best_action"], "Next action for Miro")
            self.assertNotEqual(plans[0]["next_best_action"], plans[1]["next_best_action"])
        finally:
            import sdr_pipeline_agent as agent_module
            agent_module.plan_prospect = original_plan_prospect


if __name__ == "__main__":
    unittest.main()
