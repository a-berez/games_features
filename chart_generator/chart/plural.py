from __future__ import annotations


def _team_word(n: int, forms: tuple[str, str, str]) -> str:
    """Возвращает форму слова «команда» для числа n (именительный падеж)."""
    n_abs = abs(n) % 100
    n1 = n_abs % 10
    if 11 <= n_abs <= 19:
        return forms[2]
    if n1 == 1:
        return forms[0]
    if 2 <= n1 <= 4:
        return forms[1]
    return forms[2]


def teams_nominative(n: int) -> str:
    """«1 команда», «2 команды», «5 команд»."""
    return f"{n} {_team_word(n, ('команда', 'команды', 'команд'))}"


def teams_took_question(n: int) -> str:
    """«взяла 1 команда», «взяли 3 команды» — команды как подлежащее."""
    teams = teams_nominative(n)
    if n % 10 == 1 and n % 100 != 11:
        return f"взяла {teams}"
    return f"взяли {teams}"
