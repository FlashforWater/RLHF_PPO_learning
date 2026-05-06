import unittest

from data.data_loader import split_prompt_response


class SplitPromptResponseTest(unittest.TestCase):
    def test_splits_on_last_assistant_turn(self):
        text = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: again\n\nAssistant: final"

        prompt, response = split_prompt_response(text)

        self.assertEqual(prompt, "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: again")
        self.assertEqual(response, "final")

    def test_rejects_missing_separator(self):
        with self.assertRaises(ValueError):
            split_prompt_response("\n\nHuman: no assistant turn")


if __name__ == "__main__":
    unittest.main()
