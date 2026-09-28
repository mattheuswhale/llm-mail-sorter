
import unittest

from maildigest.digest import DigestItem, sort_items


def item(priority: str, action: bool = False) -> DigestItem:
    return DigestItem(
        message_id=priority + str(action),
        sender="sender@example.com",
        subject="Subject",
        date="",
        priority=priority,
        category="other",
        action_required=action,
        deadline=None,
        summary="Summary",
        reason="Reason",
        rule="test",
        signals={},
    )


class DigestSortTests(unittest.TestCase):
    def test_priority_sorting(self):
        values = [
            item("low"),
            item("normal"),
            item("urgent"),
            item("high"),
        ]
        result = sort_items(values)
        self.assertEqual(
            [x.priority for x in result],
            ["urgent", "high", "normal", "low"],
        )

    def test_action_required_sorts_first_within_priority(self):
        values = [item("normal", False), item("normal", True)]
        result = sort_items(values)
        self.assertTrue(result[0].action_required)


if __name__ == "__main__":
    unittest.main()
