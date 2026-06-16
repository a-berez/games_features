from __future__ import annotations

import plotly.graph_objects as go

from .data import TeamSeries


def build_plotly_html(teams: list[TeamSeries], questions_count: int, title: str) -> str:
    if not teams:
        return "<p>Нет данных для построения графика</p>"

    fig = go.Figure()
    for team in teams:
        legend_name = f"{team.name} ({team.town}) - {team.total_takes}"
        fig.add_trace(
            go.Scatter(
                x=list(range(1, questions_count + 1)),
                y=team.cumulative_takes,
                mode="lines",
                name=legend_name,
                hovertemplate="Вопрос %{x}<br>Всего взято: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{title} — график взятий вопросов",
        xaxis_title="Номер вопроса",
        yaxis_title="Количество взятых вопросов",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
            font=dict(size=10),
            itemwidth=30,
            tracegroupgap=5,
        ),
        margin=dict(l=50, r=50, t=80, b=150),
        autosize=True,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Всего вопросов: {questions_count}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                borderpad=4,
                align="left",
            )
        ],
    )

    fig.update_xaxes(tickmode="linear", dtick=5, range=[0.5, questions_count + 0.5])
    fig.update_yaxes(
        range=[0, questions_count],
        tickmode="linear",
        dtick=max(1, questions_count // 10),
        tickvals=list(range(0, questions_count + 1, max(1, questions_count // 10))) + [questions_count],
    )

    config = {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False,
        "scrollZoom": True,
    }
    post_script = """
    <script>
    (function() {
        var graphDiv = document.querySelector('.plotly-graph-div');
        function updateLegendPosition() {
            if (!graphDiv || !window.Plotly) return;
            if (window.innerWidth < 768) {
                Plotly.relayout(graphDiv, {
                    'legend.orientation': 'h',
                    'legend.y': -0.2,
                    'legend.x': 0,
                    'margin.b': 150
                });
            } else {
                Plotly.relayout(graphDiv, {
                    'legend.orientation': 'v',
                    'legend.y': 1,
                    'legend.x': 1.02,
                    'margin.r': 160,
                    'margin.b': 50
                });
            }
        }
        updateLegendPosition();
        window.addEventListener('resize', updateLegendPosition);
    })();
    </script>
    """

    html = fig.to_html(full_html=False, include_plotlyjs="cdn", config=config)
    return html + post_script

