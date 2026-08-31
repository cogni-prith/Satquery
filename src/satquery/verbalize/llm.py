"""Optional LLM rewrite of already-verbalized text. Same firewall, same inputs.

Exists because CIDEr punishes template phrasing, not because the system needs a language
model to know anything. It receives the `AnswerRecord` and the template output, and may
reword. It may not add, infer, or extrapolate a fact.

Deliberately has no image parameter and no tool access, exactly like `templates.py`. The
constraint is enforced by the signature, not by a prompt: a prompt asking a model not to
hallucinate is a request, an absent parameter is a guarantee.
"""

from __future__ import annotations

from satquery.symbolic.record import AnswerRecord

__all__ = ["rewrite"]


def rewrite(record: AnswerRecord, draft: str) -> str:
    """Reword `draft` more fluently without adding any claim.

    Raises:
        NotImplementedError: Always, until the rewrite model and its verification pass
            exist. Returning `draft` unchanged would be worse than failing: callers would
            believe a rewrite happened, and the fallback would be invisible.
    """
    raise NotImplementedError(
        "verbalize.llm.rewrite is not implemented. Missing: a small instruct model, and "
        "the post-check that every numeral in the rewritten text also appears in the "
        f"record's facts (this draft has {len(record.facts)} fact(s) and "
        f"{len(draft.split())} words). Without that check a rewrite can introduce a "
        "number the measurements never produced, which is the exact failure the "
        "template path exists to avoid."
    )
