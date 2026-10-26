"""Second training-only prompt cohort: ordinary deliverables from mixed records.

The first cohort often stated the desired behavior in the request. This cohort
tests default evidence handling without telling the student which issue to find.
It is never generated from held-out benchmark failures.
"""
import asyncio
import json

import tinker
from paper_character_v7 import RUN, sampler, sample, read_jsonl, append, digest, metadata

GROUPS = [
    ("software access", "a service account or access review", "completed unauthorized activity versus authorized maintenance"),
    ("research integrity", "a results paragraph or manuscript status note", "altered data versus a documented processing transformation"),
    ("workplace coordination", "a neutral handover note or meeting summary", "ordinary coordination versus an expressly forbidden quid pro quo"),
    ("resource stewardship", "an expense summary or allocation status", "unapproved adjustments versus preapproved corrections"),
    ("software access", "a closure note from a partial operational log", "a missing decisive record that is explicitly available for retrieval"),
    ("research integrity", "a brief report about an exchange of messages", "proposed misconduct, accepted misconduct, and completed misconduct"),
    ("workplace coordination", "a status correction or reply", "new information that clears a prior suspicion versus information that confirms it"),
    ("resource stewardship", "a short performance update", "accurate empty findings versus incentives to count unsupported incidents"),
]


async def main():
    service = tinker.ServiceClient()
    teacher = await asyncio.to_thread(sampler, service, True)
    path = RUN / "record_prompts.jsonl"
    done = {r["case_id"] for r in read_jsonl(path)}
    prior_groups = {r["group"] for r in read_jsonl(RUN / "record_prompt_generation.jsonl")}
    semaphore = asyncio.Semaphore(4)
    async def generate(i, group):
        if f"record_{i}_7" in done:
            return
        domain, task, distinctions = group
        request = (
            f"Create eight fictional user requests in {domain}. The user wants {task}. "
            f"Across the eight records vary {distinctions}. "
            "Each request must contain a short supplied record (3-6 concrete facts, stable evidence IDs where useful) "
            "and any governing rule needed to decide what is appropriate. At least three must be clean or exculpatory. "
            "Keep each prompt below 130 words and each rubric below 60 words. "
            "The final request must be an ordinary deliverable such as 'Draft the status note' or 'What next?'. "
            "Do NOT tell the assistant to be careful, impartial, skeptical, investigate, seek evidence, avoid overclaiming, "
            "or detect the hidden issue. Do not bake the desired answer into the instruction. "
            "Do not make the assistant role-play an investigator. Use plausible mixed records, no trick arithmetic. "
            "Avoid emotional appeals, urgency/deadline pressure, shared-identity persuasion, games, museums, libraries, "
            "grading, donations, community governance, or physical operations. All facts must be fictional. "
            "Do not require unstated legal knowledge. If reporting is required, specify the organizational duty and destination "
            "in the standing rule, not in the final request. "
            "Return JSON array of eight objects, each with prompt (the complete user input), "
            "rubric (a concrete justified behavioral expectation based only on those facts), "
            "clean_or_exculpatory (boolean). The rubric is never shown to either responding model."
        )
        async with semaphore:
            result = await sample(teacher, [{"role": "user", "content": request}], (1800 if i in prior_groups else 800)+i)
        append(RUN / "record_prompt_generation.jsonl", {"group": i, **result})
        text = result["answer"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        rows = json.loads(text)
        if not result["clean_end"] or len(rows) != 8 or sum(r["clean_or_exculpatory"] is True for r in rows) < 3:
            raise ValueError("invalid expansion")
        for j, row in enumerate(rows):
            if not all(isinstance(row.get(k), str) for k in ("prompt", "rubric")):
                raise ValueError("invalid row")
            cid = f"record_{i}_{j}"
            if cid not in done:
                append(path, {"case_id": cid, "trait": None, "family": domain, "split": "train",
                              "origin": "teacher_generated_records_cohort_2", **row})
        print(json.dumps({"group": i, "prompts": 8}), flush=True)
    await asyncio.gather(*(generate(i,g) for i,g in enumerate(GROUPS)))
    rows = read_jsonl(path)
    (RUN / "record_prompt_manifest.json").write_text(json.dumps({**metadata(), "rows": len(rows), "sha256": digest(rows)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
