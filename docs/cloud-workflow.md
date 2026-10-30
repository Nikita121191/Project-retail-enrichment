# Проверенный облачный запуск Retail Enrichment

Документ описывает настройку и восстановление облачной среды, проверенные
28 октября 2026 года на коммите `971bfb22e96377e662191e496dfea78adee97c00`.
Это запуск inference на готовом опубликованном комплекте. Обучение для
восстановления не требуется. Общая архитектура проекта описана в [README](../README.md).

## Код и границы окружения

Для нового рабочего каталога код получают из ветки `retail-enrichment`:

```bash
cd /workspace
git clone --branch retail-enrichment --single-branch \
  https://github.com/Nikita121191/Project-retail-enrichment.git
cd /workspace/Project-retail-enrichment
git rev-parse HEAD
git status --short --branch
```

Ветка может продвинуться. Для воспроизведения описанной проверки сравните HEAD
с указанным выше SHA; при расхождении сначала изучите изменения. В уже
восстановленной среде повторный clone не нужен.

| Расположение | Происхождение и назначение |
| --- | --- |
| `README.md`, `compose.yml`, `Dockerfile`, `docker/streamlit/Dockerfile` | Файлы репозитория |
| `requirements*.txt`, `src/`, `app/`, `config/`, `dags/` | Код, зависимости и конфигурация из Git |
| `artifacts/current/`, другие подкаталоги `artifacts/` | Переданные отдельно модели и история; не входят в clone |
| `data/russian_retail.csv` | Dataset хранится в репозитории и входит в clone; локальная копия отдельно передавалась для сравнения |
| `.env` | Локальная конфигурация, создаётся отдельно; секреты не публикуются |
| `/workspace/retail-setup/` | Вспомогательные файлы только подготовленной облачной среды |
| `/workspace/retail-transfer/` | Локальные материалы переноса, распаковки и прежних проверок |
| `/workspace/work/` | Новые отчёты и временные материалы вне репозитория |

В `/workspace/retail-setup/` сохранены `install.sh`, `start.sh`,
`airflow-env.sh`, `compose.runtime.yml`, `Dockerfile.streamlit`, `wheels/`,
`airflow-venv/`, `airflow/` и `dags/retail_local.py`.
**Этих файлов нет после обычного git clone.** Приведённые ниже команды с такими
путями предназначены для восстановленной подготовленной среды. Для чистой среды
нужно отдельно подготовить эквивалентную конфигурацию и зависимости.

Сохранённый `install.sh` описывает первоначальную подготовку: создаёт окружения,
скачивает зависимости, генерирует локальную конфигурацию и DAG-адаптер, собирает
Streamlit, мигрирует БД и запускает тесты. При восстановлении его повторно не
запускали; использовали готовые зависимости, образы и `start.sh`.

## Передача и целостность артефактов

Артефакты передают отдельно по согласованному каналу. Архив, модели,
личные ссылки на скачивание, логи и отчёты запусков не добавляют в Git.
Dataset `data/russian_retail.csv` уже отслеживается Git и получается вместе с кодом.
Перед загрузкой моделей проверяют происхождение комплекта и SHA-256.

Для нового переноса можно сформировать контрольный список на стороне источника
из корня передаваемого дерева (рядом с каталогом `artifacts`):

```bash
find artifacts -type f -print0 | sort -z | xargs -0 sha256sum > artifacts.sha256
```

Передайте этот список вместе с комплектом, распакуйте в отдельный каталог и
проверьте из корня распакованного дерева:

```bash
sha256sum --check artifacts.sha256
```

Это рецепт следующего переноса: наличие `artifacts.sha256` в сохранённой среде
не предполагается. В выполненном переносе проверили CRC архива и структуру
распаковки, затем сравнили SHA-256 всех 350 файлов источника и назначения.
Источник распаковки: `/workspace/retail-transfer/extracted/artifacts`;
назначение: `/workspace/Project-retail-enrichment/artifacts`.
Существующие файлы с отличающимися хешами требуют отдельного разбора до замены.
При восстановлении сначала проверяют наличие файлов; отсутствующие модели
автоматически не скачивают и не пересоздают.

Восстановленный комплект расположен в
`/workspace/Project-retail-enrichment/artifacts/current`. В Streamlit он виден
как `/app/artifacts/current`; весь каталог `artifacts` подключён read-only.
Минимальный serving-набор:

```text
current/
  manifest.json
  domain/domain_model.joblib
  domain/domain_classes.json
  founded/founded_model.joblib
  price_category/price_category_model.joblib
  price_category/price_category_labels.json
  price_category/price_category_thresholds.json
```

Для полной штатной валидации нужны также отчёты моделей, leaderboard и остальные
файлы из `REQUIRED_FILES` в `src/validate_artifacts.py`. В восстановленном наборе
есть и `publication_evidence/` с историческими отчётами quality gate и smoke.

`manifest.json` фиксирует исходный run, время публикации, fingerprint обучающего
dataset, версии feature contract и quality policy, выбранные модели и статусы
проверок. `src/serving_contract.py` проверяет manifest и обязательные serving-файлы
до inference. Manifest не заменяет проверку SHA-256 самих файлов моделей.
Его `dataset_sha256` относится к обучающему dataset. Хеш приложенного CSV
с окончаниями строк CRLF совпал с хешем в исторических отчётах; файл из Git
использует LF и потому имеет другой побайтовый SHA-256. Сравнение подтвердило:
оба CSV содержат 2737 строк данных и 11 колонок, имена и порядок колонок,
порядок строк и все значения совпадают. После замены CRLF на LF файлы
побайтово одинаковы; BOM отсутствует в обоих.

Проверенные SHA-256:

- приложенная Windows-копия (CRLF): `2544705b3c1fd43509f3366066ebf3153b9bbfc7658b4326056d62f157f5d570`;
- файл проекта (LF): `da0d2a22751739a59542cc278b2c5aeb821b3e41f24602ac2a633fb7e211f745`.

Эта проверка подтверждает совпадение содержимого сравниваемых CSV. Smoke
проверяет работоспособность inference и не является новой оценкой качества
на независимых данных.

## Фактическое окружение

Версии прочитаны из работающего контейнера Streamlit:

| Компонент | Версия |
| --- | --- |
| Python | 3.10.22 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.3.2 |
| SciPy | 1.15.3 |
| joblib | 1.3.2 |
| XGBoost | 2.0.3 |
| Matplotlib | 3.8.4 |
| Streamlit | 1.35.0 |

В metadata моделей указан Python 3.10.20; фактическая загрузка и inference
проверены на 3.10.22. NumPy, pandas и scikit-learn совпадают с metadata.
`requirements-serving.txt` фиксирует основные пакеты, но не Python и SciPy;
таблица отражает проверенный runtime, а не гарантию будущей сборки по изменяемому
тегу `python:3.10-slim`.

PostgreSQL работает в Docker: 17.11 (образ `postgres:17`). Streamlit тоже в Docker.
Airflow 3.3.0 работает **вне Docker**, в
`/workspace/retail-setup/airflow-venv` на Python 3.10.22 с LocalExecutor.
NumPy, pandas, scikit-learn, SciPy и joblib в этом окружении имеют версии из таблицы.

Облачный `compose.runtime.yml` открывает PostgreSQL на `127.0.0.1:5432` и
подставляет вспомогательный Dockerfile Streamlit с локальным wheel-каталогом.
`airflow-env.sh` задаёт подключение к локальному PostgreSQL, пути Airflow и PATH
виртуального окружения. Значения учётных данных здесь не приводятся.
`dags/retail_local.py` адаптирует пути исходного DAG из `/opt/airflow/project`
и `/opt/airflow/config` к облачному рабочему каталогу без правок исходного DAG.

## Запуск подготовленной среды

Проверьте доступность локального Docker daemon и сохранённых файлов. Затем:

```bash
cd /workspace/Project-retail-enrichment
bash /workspace/retail-setup/start.sh
```

Скрипт использует сохранённый Docker config и `airflow-env.sh`, выполняет:

```bash
docker compose -f compose.yml \
  -f /workspace/retail-setup/compose.runtime.yml \
  up -d --no-build postgres streamlit
```

Он проверяет Airflow health и при необходимости запускает `airflow standalone`.
При восстановлении PostgreSQL и Streamlit уже были healthy. Airflow пришлось
запустить; фоновый процесс скрипта остановился после завершения команды.
Рабочий вариант — отдельная длительная терминальная сессия, в которой остаётся
foreground-процесс (использовать только если Airflow ещё не работает):

```bash
source /workspace/retail-setup/airflow-env.sh
airflow standalone
```

В этой схеме Airflow доступен на `http://127.0.0.1:8080`, Streamlit — на
`http://127.0.0.1:8501`. Порт 8082 из общего README относится к Docker-варианту
Airflow. Для доступа с пользовательского компьютера нужен поддерживаемый средой
просмотр или проброс порта; localhost в этих командах относится к облачной машине.
**Автономная работа Airflow после завершения сессии пока не проверена.**
Не запускайте training DAG для проверки восстановления.

## Проверки сервисов и готового комплекта

Из корня проекта, в отдельном терминале:

```bash
source /workspace/retail-setup/airflow-env.sh
export PYTHONDONTWRITEBYTECODE=1
docker compose -f compose.yml \
  -f /workspace/retail-setup/compose.runtime.yml \
  exec -T postgres pg_isready -U airflow -d airflow
python - <<'PY'
import json
import urllib.request
health = json.load(urllib.request.urlopen(
    'http://127.0.0.1:8080/api/v2/monitor/health', timeout=5))
assert all(health[k]['status'] == 'healthy' for k in
           ['metadatabase', 'scheduler', 'triggerer', 'dag_processor'])
assert urllib.request.urlopen(
    'http://127.0.0.1:8501/_stcore/health', timeout=5).read() == b'ok'
print('Airflow and Streamlit healthy')
PY
PYTHONPATH=src python - <<'PY'
from serving_contract import validate_published_bundle
validate_published_bundle('artifacts/current')
print('Serving contract: PASSED')
PY
```

Новые результаты сохраняйте вне `artifacts/current` и исторических запусков.
Следующий блок создаёт уникальный каталог и использует готовые модели:

```bash
mkdir -p /workspace/work
check_dir=$(mktemp -d /workspace/work/retail-check-XXXXXXXX)
python src/validate_artifacts.py \
  --domain_dir artifacts/current/domain \
  --founded_dir artifacts/current/founded \
  --price_dir artifacts/current/price_category \
  > "$check_dir/artifact-validation.log" 2>&1
python src/model_quality_gate.py \
  --training_dir artifacts/current \
  --config_path config/model_quality_thresholds.json \
  --out_path "$check_dir/model_quality_gate_report.json"
python src/smoke_inference.py \
  --csv_path data/russian_retail.csv \
  --artifacts_dir artifacts/current \
  --out_dir "$check_dir/smoke" --sample_size 5 --random_state 42
```

Для smoke используется отслеживаемый Git файл `data/russian_retail.csv`.
Если он отсутствует, следует проверить рабочую копию и сообщить о проблеме,
а не запускать обучение. Quality gate проверяет сохранённые метрики моделей.

В браузере загрузите полученный `smoke/smoke_input_without_targets.csv` в Streamlit,
нажмите «Запустить предсказание», затем «Скачать predictions.csv». Сравните число
строк и предсказания со `smoke/smoke_predictions.csv`.

## Что подтвердило восстановление опубликованной среды

В новой задаче восстановились код на ожидаемом HEAD, чистое рабочее дерево,
вспомогательные файлы настройки, окружение Airflow, dataset, текущие и исторические
артефакты, Docker-образы и контейнеры. PostgreSQL и Streamlit уже работали при
первой проверке. Это наблюдение конкретного опубликованного снимка среды,
а не гарантия сохранения любых процессов при каждой будущей публикации.

Повторная проверка дала:

- все обязательные файлы комплекта присутствуют;
- PostgreSQL принимает подключения, четыре компонента Airflow healthy,
  Streamlit отвечает `ok`, модели подключены read-only;
- serving contract и штатная валидация артефактов пройдены;
- quality gate: 17/17; новый smoke на 5 строках: 17/17;
- Chromium: загрузка CSV, предсказания и скачивание успешны; основные
  предсказания совпали со smoke, ошибок страницы нет;
- SHA-256 всех 401 проверенного исторического файла артефактов не изменились.

Новые отчёты остались отдельно в `/workspace/work/restore-check-20261028`.
Обучение, повторная загрузка и пересоздание моделей при восстановлении не
выполнялись. Для изменения только этой документации повторять обучение и полный
набор тестов не требуется.
