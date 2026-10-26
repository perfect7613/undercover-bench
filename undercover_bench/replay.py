"""Replay a logged prefix exactly before continuing a stopped engineering game."""
from copy import deepcopy
import json


class ReplayPrefixPolicy:
    def __init__(self, delegate, records):
        start = records[0]
        if start.get("kind") != "episode_start" or start.get("condition") != "B1":
            raise ValueError("Only B1 engineering prefixes are supported")
        if any(row["kind"] in ("invalid_action", "episode_end") for row in records):
            raise ValueError("Cannot turn a completed or invalid episode into a new continuation")
        self.delegate = delegate
        self.rows = [row for row in records if row["kind"] == "decision"]
        self.parent = start["episode_id"]
        self.assignments = delegate.assignments
        self.initialization = delegate.initialization
        self.model_key = delegate.model_key
        self.protocol_metadata = {**delegate.protocol_metadata, "continuation_of": self.parent,
            "replayed_decisions": len(self.rows), "independent_new_episode": False}

    def set_episode_seed(self, seed):
        self.delegate.set_episode_seed(seed)
        self.index = 0
        self.new_decisions = 0

    def decide(self, observation, legal):
        if self.index >= len(self.rows):
            response = self.delegate.decide(observation, legal)
            self.new_decisions += 1
            return response
        row = self.rows[self.index]
        if (row["observation"] != observation or row["legal"] != legal
                or row["player"] != observation["you"] or row["phase"] != observation["phase"]):
            raise ValueError(f"Replay diverged from controller state at original sequence {row['sequence']}")
        policy = (self.delegate.auditor_policy if observation["you"] == self.delegate.auditor
                  else self.delegate.policies[self.assignments[observation["you"]]])
        if policy.next_decision_seed(observation) != row["model_metadata"]["sampling_seed"]:
            raise ValueError("Replayed sampling seed differs from original prefix")
        self.index += 1
        response = {key: deepcopy(row[key]) for key in (
            "action", "rationale", "message", "evidence_ids", "raw_output", "model_emitted_reasoning",
            "model_message", "model_key", "model_metadata") if key in row}
        response["model_metadata"]["replayed_from"] = {"episode_id": self.parent, "sequence": row["sequence"]}
        return response
