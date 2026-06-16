from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from .constants import DEFAULT_ACCENT_COLOR, TEMPLATES_DIR
from .static_bundle import SCRIPTS_JS, STYLE_CSS, write_bundled_static
from .stats import QuestionStats

logger = logging.getLogger(__name__)


def render_template(name: str, **placeholders: str) -> str:
    text = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    for key, value in placeholders.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def accent_style_block(accent_color: str | None) -> str:
    color = accent_color or DEFAULT_ACCENT_COLOR
    return f"<style>:root {{ --accent-color: {html.escape(color)}; }}</style>"


def head_block(*, static_prefix: str, accent_color: str | None) -> str:
    return (
        f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet" />'
        f'{accent_style_block(accent_color)}'
        f'<link rel="stylesheet" type="text/css" href="{html.escape(static_prefix)}{STYLE_CSS}" />'
        f'<link rel="icon" type="image/png" href="/charts/favicon/favicon-96x96.png" sizes="96x96" />'
        f'<link rel="icon" type="image/svg+xml" href="/charts/favicon/favicon.svg" />'
        f'<link rel="shortcut icon" href="/charts/favicon/favicon.ico" />'
        f'<link rel="apple-touch-icon" sizes="180x180" href="/charts/favicon/apple-touch-icon.png" />'
        f'<link rel="manifest" href="/charts/favicon/site.webmanifest" />'
    )


def scripts_block(*, static_prefix: str) -> str:
    return f'<script type="text/javascript" src="{html.escape(static_prefix)}{SCRIPTS_JS}"></script>'


def copy_static_assets(static_dir: Path) -> None:
    write_bundled_static(static_dir)
    logger.debug("Собраны %s и %s в %s", STYLE_CSS, SCRIPTS_JS, static_dir)


def tournament_id_line(tournament_id: int | None) -> str:
    if not tournament_id:
        return ""
    return (
        f'<p>ID турнира: <a href="https://rating.chgk.info/tournament/{tournament_id}" '
        f'target="_blank">{tournament_id}</a></p>'
    )


def links_block_html(*, chart_href: str, table_href: str | None, crosstable_href: str | None) -> str:
    parts: list[str] = []
    if table_href:
        parts.append(f'<p>Интерактивную таблицу можно посмотреть <a href="{html.escape(table_href)}" target="_blank">тут</a></p>')
    if crosstable_href:
        parts.append(f'<p>Шахматку можно посмотреть <a href="{html.escape(crosstable_href)}" target="_blank">тут</a></p>')
    if parts:
        return "".join(parts)
    return (
        f'<p>Интерактивную таблицу можно посмотреть <a href="{html.escape(chart_href)}" target="_blank">тут</a></p>'
    )


def statistics_html(stats: QuestionStats) -> str:
    out = ['<div class="statistics-container">']

    out.append('<div class="statistics-section tombs">')
    out.append("<h3>Гробы (вопросы, которые не взяла ни одна команда)</h3>")
    if stats.tombs:
        out.append('<ul class="question-list">')
        out.extend(f"<li>Вопрос {q}</li>" for q in stats.tombs)
        out.append("</ul>")
    else:
        out.append("<p>Нет гробов</p>")
    out.append("</div>")

    out.append('<div class="statistics-section anti-tombs">')
    out.append("<h3>Антигробы (вопросы, которые взяли все команды)</h3>")
    if stats.anti_tombs:
        out.append('<ul class="question-list">')
        out.extend(f"<li>Вопрос {q}</li>" for q in stats.anti_tombs)
        out.append("</ul>")
    else:
        out.append("<p>Нет антигробов</p>")
    out.append("</div>")

    out.append('<div class="statistics-section hardest-questions">')
    out.append("<h3>Самые сложные вопросы</h3>")
    if stats.hardest:
        out.append("<ul>")
        for q in stats.hardest:
            out.append(f'<li class="question-item">Вопрос {q.question_num} — взяли {q.takes_count} команд')
            if q.teams_took and len(q.teams_took) <= 10:
                out.append(f'<div class="teams-took">Взяли: {html.escape(", ".join(q.teams_took))}</div>')
            elif q.teams_took and len(q.teams_took) > 10:
                out.append(f'<div class="teams-took">Взяли {len(q.teams_took)} команд</div>')
            out.append("</li>")
        out.append("</ul>")
    else:
        out.append("<p>Нет данных</p>")
    out.append("</div>")

    out.append('<div class="statistics-section easiest-questions">')
    out.append("<h3>Самые легкие вопросы</h3>")
    if stats.easiest:
        out.append("<ul>")
        for qnum, takes in stats.easiest:
            out.append(f"<li>Вопрос {qnum} — взяли {takes} команд</li>")
        out.append("</ul>")
    else:
        out.append("<p>Нет данных</p>")
    out.append("</div>")

    out.append('<div class="statistics-section removed-questions">')
    out.append("<h3>Снятые вопросы</h3>")
    if stats.removed:
        out.append('<ul class="question-list">')
        out.extend(f"<li>Вопрос {q}</li>" for q in stats.removed)
        out.append("</ul>")
    else:
        out.append("<p>Нет снятых вопросов</p>")
    out.append("</div>")

    out.append("</div>")
    return "".join(out)


def build_chart_page(
    *,
    page_title: str,
    teams_count: int,
    chart_html: str,
    stats_html: str,
    tournament_id: int | None,
    chart_href: str,
    table_href: str | None,
    crosstable_href: str | None,
    static_prefix: str,
    accent_color: str | None,
) -> str:
    return render_template(
        "chart.html",
        page_title=html.escape(page_title),
        teams_count=html.escape(f"{teams_count} команд"),
        chart_html=chart_html,
        statistics_html=stats_html,
        tournament_id_line=tournament_id_line(tournament_id),
        links_block=links_block_html(chart_href=chart_href, table_href=table_href, crosstable_href=crosstable_href),
        head_block=head_block(static_prefix=static_prefix, accent_color=accent_color),
        scripts_block=scripts_block(static_prefix=static_prefix),
    )


def build_table_page(
    *,
    page_title: str,
    csv_href: str,
    tournament_id: int | None,
    chart_href: str,
    crosstable_href: str | None,
    static_prefix: str,
    accent_color: str | None,
) -> str:
    return render_template(
        "table.html",
        page_title=html.escape(page_title),
        csv_href=html.escape(csv_href),
        tournament_id_line=tournament_id_line(tournament_id),
        links_block=links_block_html(chart_href=chart_href, table_href=None, crosstable_href=crosstable_href),
        head_block=head_block(static_prefix=static_prefix, accent_color=accent_color),
        scripts_block=scripts_block(static_prefix=static_prefix),
    )


def build_index_page(
    *,
    page_title: str,
    chart_href: str,
    table_href: str | None,
    crosstable_href: str | None,
    tournament_id: int | None,
    static_prefix: str,
    accent_color: str | None,
) -> str:
    table_link = f'<li><a href="{html.escape(table_href)}" target="_blank">Таблица</a></li>' if table_href else ""
    crosstable_link = (
        f'<li><a href="{html.escape(crosstable_href)}" target="_blank">Шахматка</a></li>' if crosstable_href else ""
    )
    return render_template(
        "index.html",
        page_title=html.escape(page_title),
        chart_href=html.escape(chart_href),
        table_link=table_link,
        crosstable_link=crosstable_link,
        tournament_id_line=tournament_id_line(tournament_id),
        head_block=head_block(static_prefix=static_prefix, accent_color=accent_color),
    )


def build_crosstable_tab_markup(sheets: Sequence[Any]) -> tuple[str, str, str]:
    tab_buttons: list[str] = []
    tab_panels: list[str] = []
    for index, sheet in enumerate(sheets):
        active = " active" if index == 0 else ""
        tab_buttons.append(
            f'<button type="button" class="tab{active}" data-tab="{html.escape(sheet.key)}">'
            f"{html.escape(sheet.label)}</button>"
        )
        tab_panels.append(
            f'<div class="tab-panel{active}" data-tab="{html.escape(sheet.key)}">'
            '<div class="matrix-mount"></div>'
            "</div>"
        )
    return "".join(tab_buttons), "".join(tab_panels), html.escape(sheets[0].key)


def build_crosstable_page(
    *,
    page_title: str,
    tournament_id: int | None,
    data_href: str,
    tab_buttons: str,
    tab_panels: str,
    active_tab: str,
    chart_href: str,
    table_href: str | None,
    static_prefix: str,
    accent_color: str | None,
) -> str:
    tournament_line = (
        f'<p class="subtitle">Шахматка • турнир {tournament_id}</p>'
        if tournament_id is not None
        else '<p class="subtitle">Шахматка</p>'
    )
    links = []
    links.append(f'<p>График можно посмотреть <a href="{html.escape(chart_href)}" target="_blank">тут</a></p>')
    if table_href:
        links.append(f'<p>Интерактивную таблицу можно посмотреть <a href="{html.escape(table_href)}" target="_blank">тут</a></p>')
    return render_template(
        "crosstable.html",
        page_title=html.escape(page_title),
        tournament_line=tournament_line,
        tab_buttons=tab_buttons,
        tab_panels=tab_panels,
        active_tab=active_tab,
        data_href=html.escape(data_href),
        links_block="".join(links),
        tournament_id_line=tournament_id_line(tournament_id),
        head_block=head_block(static_prefix=static_prefix, accent_color=accent_color),
        scripts_block=scripts_block(static_prefix=static_prefix),
    )
