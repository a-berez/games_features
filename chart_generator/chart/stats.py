from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionHardness:
    question_num: int
    takes_count: int
    teams_took: list[str]


@dataclass(frozen=True)
class QuestionStats:
    tombs: list[int]
    anti_tombs: list[int]
    hardest: list[QuestionHardness]
    easiest: list[tuple[int, int]]  # (question_num, takes_count)
    removed: list[int]


def calculate_questions_statistics(teams: list, questions_count: int) -> QuestionStats:
    takes_by_question = [0] * questions_count
    removed_questions: set[int] = set()

    for team in teams:
        mask = team.mask
        for i, char in enumerate(mask):
            if char == "1":
                takes_by_question[i] += 1
            elif char == "X":
                removed_questions.add(i + 1)

    teams_count = len(teams)
    tombs = [i + 1 for i, takes in enumerate(takes_by_question) if takes == 0 and (i + 1) not in removed_questions]
    anti_tombs = [
        i + 1
        for i, takes in enumerate(takes_by_question)
        if takes == teams_count and (i + 1) not in removed_questions
    ]

    takes_with_index = [
        (takes, i + 1)
        for i, takes in enumerate(takes_by_question)
        if takes > 0 and (i + 1) not in removed_questions
    ]
    takes_with_index.sort()
    hardest: list[QuestionHardness] = []
    for takes, qnum in takes_with_index[:5]:
        teams_took = [t.name for t in teams if len(t.mask) >= qnum and t.mask[qnum - 1] == "1"]
        hardest.append(QuestionHardness(question_num=qnum, takes_count=takes, teams_took=teams_took))

    takes_with_index = [
        (takes, i + 1)
        for i, takes in enumerate(takes_by_question)
        if takes < teams_count and (i + 1) not in removed_questions
    ]
    takes_with_index.sort(reverse=True)
    easiest = [(qnum, takes) for takes, qnum in takes_with_index[:5]]

    return QuestionStats(
        tombs=tombs,
        anti_tombs=anti_tombs,
        hardest=hardest,
        easiest=easiest,
        removed=sorted(removed_questions),
    )

