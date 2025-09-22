import re
from types import SimpleNamespace
from typing import Any, Callable

import verifiers as vf
from verifiers.types import ChatMessage, Messages


class CustomParser(vf.Parser):
    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        super().__init__(extract_fn=extract_fn)

    def parse(self, text: str, strip: bool = True) -> Any:
        """
        Parse the XML string and return an object with think, review, and comments attributes.
        """
        results = {"think": None, "review": None, "comments": []}

        # Extract <think> content
        think_pattern = r"<think>\s*(.*?)\s*</think>"
        think_match = re.search(think_pattern, text, re.DOTALL)
        if think_match:
            results["think"] = think_match.group(1).strip() if strip else think_match.group(1)

        # Extract <review> content
        review_pattern = r"<review>\s*(.*?)\s*</review>"
        review_match = re.search(review_pattern, text, re.DOTALL)
        if review_match:
            review_content = review_match.group(1).strip() if strip else review_match.group(1)
            results["review"] = review_content

            # Extract all <comment> tags within the review content
            comment_pattern = r"<comment>\s*(.*?)\s*</comment>"
            comment_matches = re.findall(comment_pattern, review_content, re.DOTALL)
            results["comments"] = [comment.strip() for comment in comment_matches if comment.strip()]

        return SimpleNamespace(**results)

    def parse_answer(self, completion: Messages) -> list[str]:
        """Extract the list of comments from a completion."""
        if isinstance(completion, str):
            parsed = self.parse(completion)
            return parsed.comments if hasattr(parsed, "comments") else []
        else:
            for msg in reversed(self.get_assistant_messages(completion)):
                assert "content" in msg
                content = str(msg["content"])
                parsed = self.parse(content)
                if hasattr(parsed, "comments"):
                    return parsed.comments
        return []

    def get_format_str(self) -> str:
        return (
            "<think>\n"
            "...\n"
            "</think>\n\n"
            "<review>\n"
            "<comment>...</comment>\n"
            "<comment>...</comment>\n"
            "</review>\n\n"
            "Note: If no issues found, leave <review></review> empty."
        )

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks if messages follow the expected format.
        """

        def format_reward_func(completion: list[ChatMessage], **kwargs) -> float:
            model_messages = self.get_assistant_messages(completion)
            if not model_messages:
                return 0.0

            format_scores = []
            for msg in model_messages:
                assert "content" in msg
                content = str(msg["content"])

                score = 0.0

                # Check for <think> tags (40% of score)
                if re.search(r"<think>.*?</think>", content, re.DOTALL):
                    score += 0.4

                # Check for <review> tags (40% of score)
                if re.search(r"<review>.*?</review>", content, re.DOTALL):
                    score += 0.4

                # Check that message starts with <think> (10% of score)
                if content.strip().startswith("<think>"):
                    score += 0.1

                # Check that message ends with </review> (10% of score)
                if content.strip().endswith("</review>"):
                    score += 0.1

                format_scores.append(score)

            return sum(format_scores) / len(format_scores) if format_scores else 0.0

        return format_reward_func
