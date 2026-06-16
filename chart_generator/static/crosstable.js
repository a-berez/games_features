(function () {
  const CELL_CLASSES = ['x', 'y', 'z', 'w'];
  let DATA = null;
  let activeTab = document.body.dataset.initialTab || 'all';
  const renderedTabs = new Set();

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function formatPosition(position) {
    return position === '' || position === null || position === undefined ? '' : String(position);
  }

  function renderCell(values) {
    return values.map((value, index) => {
      const part = `<span class="n n-${CELL_CLASSES[index]}">${value}</span>`;
      return index < values.length - 1 ? part + '<span class="sep">•</span>' : part;
    }).join('');
  }

  function renderMatrixTable(teams, matrix) {
    const headerCells = [
      '<th class="sticky-corner col-place">место</th>',
      '<th class="sticky-corner col-team">команда</th>',
    ];
    teams.forEach((team) => {
      const name = escapeHtml(team.name);
      headerCells.push(`<th class="team-header" title="${name}">${name}</th>`);
    });

    const bodyRows = teams.map((teamA, rowIndex) => {
      const cells = [
        `<td class="sticky-col col-place">${escapeHtml(formatPosition(teamA.position))}</td>`,
        `<td class="sticky-col col-team" title="${escapeHtml(teamA.name)}">`
          + `<span class="team-name">${escapeHtml(teamA.name)}</span></td>`,
      ];
      matrix[rowIndex].forEach((cell) => {
        if (cell === null) {
          cells.push('<td class="diagonal">×</td>');
        } else {
          cells.push(`<td>${renderCell(cell)}</td>`);
        }
      });
      return `<tr>${cells.join('')}</tr>`;
    });

    return (
      '<div class="table-wrap">'
      + '<table class="matrix">'
      + `<thead><tr>${headerCells.join('')}</tr></thead>`
      + `<tbody>${bodyRows.join('')}</tbody>`
      + '</table></div>'
    );
  }

  function renderMatrixTab(tabKey) {
    if (renderedTabs.has(tabKey)) {
      return;
    }
    const panel = document.querySelector(`#matrix-view .tab-panel[data-tab="${tabKey}"]`);
    if (!panel || !DATA) {
      return;
    }
    const mount = panel.querySelector('.matrix-mount');
    mount.innerHTML = renderMatrixTable(DATA.teams, DATA.sheets[tabKey]);
    renderedTabs.add(tabKey);
  }

  function populateTeamSelect() {
    const select = document.getElementById('team-select');
    const options = ['<option value="">Вся таблица</option>'];
    DATA.teams.forEach((team, index) => {
      const name = escapeHtml(team.name);
      options.push(`<option value="${index}">${name}</option>`);
    });
    select.innerHTML = options.join('');
    select.disabled = false;
  }

  function renderTeamView(teamIndex) {
    const tbody = document.getElementById('team-body');
    const matrix = DATA.sheets[activeTab];
    const rows = [];
    DATA.teams.forEach((opponent, index) => {
      if (index === teamIndex) {
        return;
      }
      const values = matrix[teamIndex][index];
      const name = escapeHtml(opponent.name);
      rows.push(
        `<tr>
          <td>${formatPosition(opponent.position)}</td>
          <td title="${name}">${name}</td>
          <td>${renderCell(values)}</td>
        </tr>`
      );
    });
    tbody.innerHTML = rows.join('');
  }

  function updateView() {
    const teamSelect = document.getElementById('team-select');
    const matrixView = document.getElementById('matrix-view');
    const teamView = document.getElementById('team-view');
    const legendMatrix = document.getElementById('legend-matrix');
    const legendTeam = document.getElementById('legend-team');
    const selected = teamSelect.value;
    if (selected === '') {
      matrixView.classList.remove('hidden');
      teamView.classList.remove('active');
      legendMatrix.classList.remove('hidden');
      legendTeam.classList.add('hidden');
    } else {
      matrixView.classList.add('hidden');
      teamView.classList.add('active');
      legendMatrix.classList.add('hidden');
      legendTeam.classList.remove('hidden');
      renderTeamView(Number(selected));
    }
  }

  function bindEvents() {
    document.getElementById('team-select').addEventListener('change', updateView);

    document.getElementById('tabs').addEventListener('click', (event) => {
      const button = event.target.closest('.tab');
      if (!button) {
        return;
      }
      activeTab = button.dataset.tab;
      document.querySelectorAll('.tab').forEach((tab) => {
        tab.classList.toggle('active', tab.dataset.tab === activeTab);
      });
      document.querySelectorAll('#matrix-view .tab-panel').forEach((panel) => {
        panel.classList.toggle('active', panel.dataset.tab === activeTab);
      });
      renderMatrixTab(activeTab);
      const selected = document.getElementById('team-select').value;
      if (selected !== '') {
        renderTeamView(Number(selected));
      }
    });
  }

  function showError(message) {
    const status = document.getElementById('load-status');
    status.textContent = message;
    status.classList.add('load-error');
  }

  async function init() {
    const dataHref = document.body.dataset.href;
    if (!dataHref) {
      showError('Не указан файл данных.');
      return;
    }

    try {
      const response = await fetch(dataHref);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      DATA = await response.json();
    } catch (error) {
      showError(
        'Не удалось загрузить данные. Откройте страницу через веб-сервер '
        + `(файл ${dataHref}).`
      );
      console.error(error);
      return;
    }

    document.getElementById('load-status').hidden = true;
    populateTeamSelect();
    bindEvents();
    renderMatrixTab(activeTab);
  }

  init();
})();
