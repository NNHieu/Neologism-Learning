"""Prompt templates that place the neologism in natural language.

`{neo}` is the neologism token (e.g. "~singlesentence"); `{instruction}` is the
underlying user request. The default template matches the paper's main setup;
the extra templates (Table 10) are used for the more robust multi-template runs.
"""

DEFAULT_TEMPLATE = "{instruction} Give me a {neo} answer."

# A subset of the paper's robustness templates (Table 10).
MULTI_TEMPLATES = [
    "{instruction} Give me a {neo} answer.",
    "{instruction} Give me an answer which is {neo}.",
    "Generate a {neo} response to the following:\n\n{instruction}",
    "{instruction}\n\nI would like you to use a {neo} style when responding, thanks!",
    "Craft a reply to the following that embodies the concept of {neo}:\n\n{instruction}",
    "{instruction}\n\nPlease make sure provide {neo} answers only!",
]


def render(template: str, instruction: str, neo: str) -> str:
    return template.format(instruction=instruction.strip(), neo=neo)
