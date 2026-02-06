import random
from collections import defaultdict
from typing import Any


def ema_update(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


class CurriculumPolicy:
    def __init__(
        self,
        alpha: float,
        reasoning_types: list[str],
        difficulty_levels: list[str],
        task_types: list[str],
        epsilon: float = 0.15,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.reasoning_types = reasoning_types
        self.difficulty_levels = difficulty_levels
        self.task_types = task_types
        self.epsilon = epsilon
        self.random = random.Random(seed)
        self.global_ema: float | None = None
        self.bucket_ema: dict[str, dict[str, float]] = {
            "reasoning_type": defaultdict(lambda: 0.5),
            "difficulty": defaultdict(lambda: 0.5),
            "task_type": defaultdict(lambda: 0.5),
        }
        self.combo_ema: dict[tuple[str, str, str], float] = defaultdict(lambda: 0.5)

    def replay(self, history: list[dict[str, Any]]) -> None:
        for row in history:
            self.update(row)

    def update(self, record: dict[str, Any]) -> None:
        score = float(record["evaluation"]["score"])
        self.global_ema = ema_update(self.global_ema, score, self.alpha)

        profile = record.get("question_profile", {})
        reasoning = profile.get("reasoning_type")
        difficulty = profile.get("difficulty")
        task_type = profile.get("task_type")

        if reasoning:
            prev = self.bucket_ema["reasoning_type"][reasoning]
            self.bucket_ema["reasoning_type"][reasoning] = ema_update(prev, score, self.alpha)
        if difficulty:
            prev = self.bucket_ema["difficulty"][difficulty]
            self.bucket_ema["difficulty"][difficulty] = ema_update(prev, score, self.alpha)
        if task_type:
            prev = self.bucket_ema["task_type"][task_type]
            self.bucket_ema["task_type"][task_type] = ema_update(prev, score, self.alpha)
        if reasoning and difficulty and task_type:
            key = (task_type, reasoning, difficulty)
            prev = self.combo_ema[key]
            self.combo_ema[key] = ema_update(prev, score, self.alpha)

    def sample_target(self) -> dict[str, str]:
        return {
            "reasoning_type": self._sample_dimension("reasoning_type", self.reasoning_types),
            "difficulty": self._sample_dimension("difficulty", self.difficulty_levels),
            "task_type": self._sample_dimension("task_type", self.task_types),
        }

    def _sample_dimension(self, name: str, choices: list[str]) -> str:
        if self.random.random() < self.epsilon:
            return self.random.choice(choices)

        scores = [self.bucket_ema[name][choice] for choice in choices]
        weaknesses = [max(1e-6, 1.0 - score) for score in scores]
        return self.random.choices(choices, weights=weaknesses, k=1)[0]

    def sample_buckets(
        self,
        buckets: list[tuple[str, str, str]],
        n: int,
        difficulty_prior: dict[str, float] | None = None,
        gamma: float = 1.0,
    ) -> list[tuple[str, str, str]]:
        if n <= 0:
            return []
        if self.random.random() < self.epsilon:
            return self.random.sample(buckets, k=min(n, len(buckets)))

        weights: list[float] = []
        for task_type, reasoning, difficulty in buckets:
            base = max(1e-6, 1.0 - self.combo_ema[(task_type, reasoning, difficulty)])
            prior = 1.0 if not difficulty_prior else float(difficulty_prior.get(difficulty, 1.0))
            weights.append((base**gamma) * prior)
        return self.random.choices(buckets, weights=weights, k=min(n, len(buckets)))
