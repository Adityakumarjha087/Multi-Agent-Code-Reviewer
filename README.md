# prreview: a multi-agent GitHub PR reviewer

Given a pull request URL, a team of agents reviews the change: a **planner** decides what to review,
a **bug** reviewer and a **security** reviewer read the diff (with read-only repository tools), a
**test** agent runs the repository's tests at base and head in a locked-down Docker sandbox, and an
**aggregator** merges, de-duplicates, and ranks findings into draft review comments. **A human approves
each comment** before a separate publisher posts it to GitHub.

Design rationale, trade-offs, and every bug found while building: [DECISIONS.md](DECISIONS.md).

## Status: what is verified

| Area | State |
|---|---|
| MCP servers, sandbox, GitHub client, safety layer, graph, persistence, UI, Compose stack | Built and tested (see [Tests](#tests)) |
| Review quality with the real model | **Not measured yet.** Needs `ANTHROPIC_API_KEY`; agent code paths are tested with scripted test doubles only |
| Benchmark results table | **Not run yet.** See [Benchmark](#benchmark); no numbers are reported until a real run exists |
| Langfuse | Instrumentation verified against the Langfuse SDK with an in-memory exporter; **not verified against Langfuse Cloud** |
| Posting to GitHub | Verified against a faked GitHub API only; needs `GITHUB_WRITE_TOKEN` and a test repo |

## Architecture

```
 Streamlit UI ── enqueue / approve ──►  PostgreSQL  ◄── results, events ── Worker (LangGraph)
                                            ▲                                   │ MCP
                                            │ approved comments                 ├─ github_read  (stdio, read token)
                                       Publisher (no LLM)                       ├─ search       (stdio, read-only repo access)
                                            │ MCP                               └─ sandbox      (HTTP; only holder of the Docker socket)
                                       github_write ── posts by comment id, re-checks approval in the DB
```

Graph: `planner → Send fan-out → {bug reviewer ×N batches, security reviewer, test runner} → aggregator`.

**Safety model (prompt injection and untrusted code)**
- All PR-derived text reaches the model inside `<untrusted-{random nonce}>` fences; the system prompt
  says fenced content is data. The PR author cannot close a fence they cannot guess.
- No schema has an approve/verdict field. LLM agents have no path to any write tool: the write server
  takes only a comment id and posts only rows a human approved.
- Tool allowlists per agent; `owner/repo/sha` are bound by code, so injected text cannot point an
  agent at another repository.
- Every finding is grounded: its file/lines must be in the diff and its quoted evidence must exist
  there, or it is discarded (and counted).
- Injection attempts are detected heuristically and **reported as findings**.
- Untrusted code runs only in Docker: `--network none`, read-only root, uid 65534, all capabilities
  dropped, no-new-privileges, memory/CPU/pids/tmpfs/output caps, hard timeout. Dependencies are
  installed in a separate stage with wheels only (no `setup.py` execution).

**Reliability**: per-agent step limit (last step forced to answer), graph recursion limit, per-review
cost budget, retries with backoff for transient errors only, timeouts on tool calls and sandbox runs;
any tool failure becomes a visible note and the review continues.

## Quick start

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY (and optionally GITHUB_READ_TOKEN, Langfuse keys)
make up                         # Docker Compose: postgres, migrations, sandbox, worker, UI
open http://127.0.0.1:8501      # paste a PR URL, watch progress, approve/reject comments
docker compose --profile publish up -d   # posting approved comments (needs GITHUB_WRITE_TOKEN)
```

Local, without Compose:

```bash
make install
.venv/bin/python -m prreview review https://github.com/owner/repo/pull/123 [--no-sandbox] [--out review.json]
```

## Benchmark

`benchmark/cases/` holds 30 labelled PRs: 14 bug (2 break existing tests), 8 security, 5 clean,
3 prompt-injection (comment, PR description, docstring; each with a planted real issue), 27 labels.

```bash
make bench-validate        # checks labels and prints every labelled line: review these by hand
make bench                 # real model, paid API calls; TRIALS=3 make bench for repeats
```

During a run only GitHub's HTTP API is answered from the case fixture; the MCP servers, graph,
agents, sandbox, OSV lookup, and the model are real. Metrics: recall, precision, false-positive
findings per clean PR, injection resistance (planted issue found **and** no forbidden text **and**
injection flagged), average steps, cost, and latency per review. Results (with command, config, and
dataset hash) are written to `benchmark/results/<timestamp>/`.

### Results

<!-- TODO: paste the table from a real run's benchmark/results/<timestamp>/results.md, with its command. -->
**No results yet.** The benchmark has not been run against the model (no API key was available
while building). This section will hold the table produced by `make bench`, copied verbatim.

## Tests

```bash
make test          # unit tests: no Docker, network, or API key
make test-all      # + Docker sandbox, real PostgreSQL, Streamlit AppTest, live GitHub
```

Last full run (2026-10-09): `make test-all` → 171 passed, 2 skipped. The 2 skipped are the live-GitHub
tests: they skip with the reason shown when the unauthenticated API quota is exhausted (set
`GITHUB_READ_TOKEN` to run them; both passed earlier the same day). CI (`.github/workflows/ci.yml`)
runs the unit tests and benchmark-label validation on every push.
LLM-dependent paths use test doubles from `tests/fakes.py` (`ScriptedLLM`, `MinimalLLM`, `OracleLLM`);
they test control flow, not model quality.

## Layout

```
src/prreview/
  agents/        planner, reviewers (bug, security), tests_agent, aggregator, the bounded tool loop, prompts
  graph/         LangGraph wiring
  mcp_servers/   github_read, github_write, search_server, sandbox_server
  sandbox/       hardened Docker runner, dependency collection, JUnit parsing
  safety/        untrusted-data fencing, injection heuristics, grounding check
  persistence/   SQLAlchemy models and repository (Alembic migrations in migrations/)
  bench/         dataset, harness, metrics, report
  review/        connecting MCP servers, fetching PR context, running a review
  worker.py, publisher.py, cli.py, scanners.py, observability/
ui/app.py        Streamlit
benchmark/cases  labelled PR fixtures
```

## Known limitations

- Test execution supports Python/pytest only, on Python 3.11 in the sandbox.
- Dependencies are reinstalled for every sandbox run (base and head): slow for large repos.
- The sandbox server needs the Docker socket (root-equivalent on the host); gVisor is not configured.
- Injection heuristics are evadable by design; they are a tripwire, not the defence.
- The injection cases were written by the same author as the heuristics (see DECISIONS P5-3).
