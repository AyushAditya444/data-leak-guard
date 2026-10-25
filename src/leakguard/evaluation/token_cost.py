"""Does masking change what you pay the LLM? Tokens before and after masking.

A placeholder like [AADHAAR_1] can tokenise to more or fewer tokens than the value it
replaces ("2345 6789 0124"), and LLM providers bill per token. This counts both over the
evaluation messages, as the guard would forward them.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable

from leakguard.detect.engine import Detector
from leakguard.evaluation.samples import Sample
from leakguard.masking import mask_messages
from leakguard.policy import Policy

ENCODING = "o200k_base"  # OpenAI's current models and gpt-oss


def o200k_counter() -> Callable[[str], int]:
    import tiktoken  # [perf] extra only

    encoding = tiktoken.get_encoding(ENCODING)
    return lambda text: len(encoding.encode(text))


def token_cost(samples: Iterable[Sample], detect: Detector, policy: Policy,
               count: Callable[[str], int]) -> dict:
    """Each sample is masked as one user message. Blocked messages are never sent, so they
    are left out (and counted). `by_type` compares each value with its placeholder
    tokenised on their own: in a sentence, tokens can merge across the boundary."""
    messages = blocked = unchanged = before = after = 0
    values: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for sample in samples:
        result = mask_messages([{"role": "user", "content": sample.text}], policy, detect)
        if result.blocked:
            blocked += 1
            continue
        masked = result.messages[0]["content"]
        messages += 1
        unchanged += masked == sample.text
        before += count(sample.text)
        after += count(masked)
        for placeholder, value in result.mapping.placeholders().items():
            entity = placeholder[1:-1].rsplit("_", 1)[0]  # "[BANK_ACCOUNT_2]" -> BANK_ACCOUNT
            values[entity].append((count(value), count(placeholder)))
    return {
        "messages": messages, "blocked": blocked, "unchanged": unchanged,
        "tokens_before": before, "tokens_after": after,
        "mean_change": (after - before) / messages if messages else 0.0,
        "by_type": {entity: {"values": len(pairs),
                             "value_tokens": sum(v for v, _ in pairs) / len(pairs),
                             "placeholder_tokens": sum(p for _, p in pairs) / len(pairs)}
                    for entity, pairs in sorted(values.items())},
    }
