# Design Decisions

Each entry: the decision, why, and what it costs. Newest phase at the bottom.

---

## Phase 0 — Design (approved 2026-10-09)

The user approved all recommendations in the design proposal.

| ID | Decision | Why | Trade-off |
|----|----------|-----|-----------|
| D1 | **LangGraph for orchestration; per-agent tool loop is our own code** | Fan-out (`Send`), state reducers, Postgres checkpointer come for free; owning the inner loop keeps step counting, budgets, and tool allowlists explicit and auditable | LangGraph API churn; mitigated by using only core graph primitives (APIs to be verified at Phase 3 against the installed version) |
| D2 | **Agents share typed findings, not conversations** | Smaller contexts (cost), and a prompt injection that reaches one agent cannot propagate via shared chat history | Specialists can't build on each other's reasoning |
| D3 | **Planner is part rules, part LLM** — security review always runs; tests run iff a test config exists | Injected text can't remove a mandatory review task; stable step counts | Less "agentic" |
| D4 | **Schema validation + grounding check** on every LLM output (file/line must be in the diff, quoted evidence must exist) | Removes hallucinated line numbers; GitHub only accepts comments on diff lines anyway | Can drop real findings about unchanged code |
| D5 | **Two-stage sandbox**: install with network (wheels only), run with `--network none` | Running tests needs dependencies; `--only-binary :all:` means no `setup.py` executes during install | Install stage has egress; sdist-only packages can't be installed |
| D6 | **Test agent compares base vs head**, reports only newly failing tests | Only honest way to say a failure is "related to the PR" | Doubles sandbox time |
| D7 | **Prompt-injection defense in layers**: untrusted-data delimiters, no `approve` field in any schema, per-agent tool allowlists, grounding check, injection heuristics reported as findings, human approval | No single layer is reliable alone | More components to test |
| D8 | **Security agent = scanners + LLM triage** | Scanners give recall, the LLM gives precision and cross-function reasoning | More dependencies (to verify in Phase 2/3) |
| D9 | **UI never runs the graph**; worker process + Postgres job table; a separate non-LLM publisher posts approved comments | Streamlit reruns are a bad fit for multi-minute jobs; approvals can be hours later; the LLM process can't reach the write tool | One extra process |
| D10 | **Benchmark**: seeded PRs in an own repo + clean PRs + ≥3 injection PRs; precision/recall + false positives per clean PR; N trials | Reproducible labels; false-positive rate needs negatives; LLMs are non-deterministic | Seeded bugs are less realistic than real ones |
| D11 | **Reliability**: per-agent step cap, overall recursion limit, per-review cost budget, retries only for transient errors | A step cap alone doesn't stop one huge call; retrying 4xx only wastes money | — |
| D12 | **Hard caps on PR size**; files over the cap are listed as "not reviewed" | Bounded cost per review | Large PRs get partial reviews |
| Q2 | **LLM: Claude** (Sonnet for specialists, Haiku for planner/aggregator) | Native structured output and tool use | Single-provider (thin interface, no multi-provider layer) |
| Q3 | **Test execution: Python/pytest only in v1** | Each language needs its own runner image | Other languages get static review only |
| Q6 | **Langfuse Cloud**, not self-hosted | Self-hosted v3 adds several services to Compose (to verify when we get there) | Traces leave the machine |

---

## Phase 1 — MCP tools + sandbox (2026-10-09)

### P1-1. MCP Python SDK is **v2.3.0**, not 1.x
`pip install mcp` resolved to 2.3.0. v2 renamed `FastMCP` → `mcp.server.mcpserver.MCPServer`
(importing `mcp.server.fastmcp` raises `ModuleNotFoundError` pointing to the migration guide).
Verified from source and with a probe script:
- `@server.tool()` + a Pydantic return type → `structured_content` and an `output_schema`.
- Raising `ToolError` → `is_error=True` with our message. Any other exception → `is_error=True`
  with a generic message (traceback only in the server log).
- `mcp.Client(server_instance)` connects in-process — used for tests through the real protocol path.
- `ToolAnnotations` fields are snake_case (`read_only_hint`, …).

**Consequence:** `tool_errors()` converts an explicit allowlist of *expected* exceptions into
`ToolError` so agents see the real reason ("GitHub API 404: Not Found"). Unexpected exceptions are
deliberately not caught: they're bugs and should be loud.

### P1-2. Four MCP servers, split by privilege
| Server | Holds | Tools |
|---|---|---|
| `github_read` | read token | `get_pull_request`, `list_changed_files`, `get_file_content` |
| `github_write` | **write token (only here)** | `post_review_comment` (later replaced by `post_approved_comment(comment_id)`, see P4-1) |
| `search_server` | read token | `list_files`, `search_code`, `read_file` |
| `sandbox_server` | Docker socket, read token | `run_tests`, `run_command` |

Agents will connect only to the read/search/sandbox servers; the write server is for the
publisher. Separate read/write tokens are separate env vars (`GITHUB_READ_TOKEN`, `GITHUB_WRITE_TOKEN`).

### P1-3. Sandbox file transfer: volumes, not `docker cp` into tmpfs
**What I tried first and why it failed (actual errors):**
1. `put_archive` into a tmpfs before start → `404 Could not find the file /workspace`
   (tmpfs only exists while running).
2. `put_archive` after start with `read_only=True` → `400 container rootfs is marked read-only`
   (the archive API refuses writes even into a tmpfs mount).

**Chosen pattern:** create a Docker-managed volume → fill it with `put_archive` via a container that
is *created but never started* → mount it **read-only** at `/src` in the sandbox → `cp` into a
size-capped tmpfs `/workspace`. Using daemon-managed volumes (not host bind mounts) also avoids
the host-path mismatch that breaks bind mounts when the sandbox server itself runs in a container (Phase 7).

### P1-4. Exec model + watchdog timeout
PID 1 is `sleep`; commands run via `exec`. Reasons, each verified:
- Docker's archive API **cannot read tmpfs either** (`get_archive` → 404 for a file that `cat` sees),
  so the JUnit report is read with a second `exec` (`head -c <cap>`), separate from test output,
  so test code can't spoof it via stdout markers.
- `exec` has no timeout; a watchdog thread kills the container, and the in-flight exec returns 137
  (measured: ~0.1 s after the kill). Since *we* killed it, `timed_out` is certain, not inferred.
- 137 **without** our timeout ⇒ reported as `killed_by_signal_9` (almost always OOM).
- Output is streamed and capped; excess is drained and discarded (an unread pipe would block the
  process until timeout instead of letting it finish).

### P1-5. Container hardening (all verified by integration tests)
`network_mode=none` (run stage), `read_only=True`, `user=65534:65534`, `cap_drop=ALL`,
`no-new-privileges`, `mem_limit` = `memswap_limit` (no swap), `nano_cpus`, `pids_limit`, tmpfs size caps,
image pinned by digest. Measured outcomes: socket connect → `Network is unreachable`; write to
`/etc` → `Read-only file system`; 600 MB alloc at 256 MB limit → exit 137; tmpfs fill → `No space left`;
fork loop → `Cannot fork`.

**Known limitations (not solved in Phase 1):**
- Volume size can't be capped on Docker Desktop's overlay2. Mitigation: post-install `du` check
  rejects deps over `SANDBOX_MAX_DEPS_MB`; source size is capped at extraction.
- The sandbox server needs the Docker socket (≈ root on the host). It exposes only fixed operations;
  callers can't set network, mounts, or limits. gVisor isn't installed here (`runc` only).
- `run_tests` does a full install per call (no dependency cache). Base+head = two installs.
  Revisit if benchmark latency is too high.
- Dependencies install for the sandbox image's Python (3.11), not the repo's declared version.

### P1-6. Treat repository files as hostile input
- **Tarball extraction:** Python's `filter="data"` (blocks `..`, absolute paths, links escaping the
  destination) + file-count and total-size caps checked *before* writing + atomic rename so a crash
  never leaves a half-extracted snapshot that looks valid.
- **Requirements files:** only plain PEP 508 specifiers reach pip. `--index-url`/`--extra-index-url`,
  `-e`, `-r`, `-c`, URLs and paths are skipped and reported (they could point pip at an attacker's index).
- **Test paths:** rejected if they start with `-` (pytest would read them as options), are absolute,
  or contain `..`.
- **JUnit XML** (written by untrusted code) parsed with `defusedxml` (entity-expansion bombs blocked).
- **Search is literal-only.** No regex: Python's `re` has no timeout, so a crafted pattern could hang
  the server (ReDoS).

### P1-7. GitHub client
- Endpoints and the `X-GitHub-Api-Version: 2026-03-10` header checked against docs.github.com and a
  live request (200).
- Retries (tenacity 9.2.1): only transport errors, 408/429/5xx. 404/401/422 fail fast. An exhausted
  rate limit (`x-ratelimit-remaining: 0`) fails fast with the reset time (waiting up to an hour isn't useful).
- **Posting a comment is never retried**: if the response is lost after GitHub created the comment,
  a retry would double-post. Better to surface the error to the human.
- Found while reading tenacity's source: `initial=` is deprecated (→ `multiplier=`), and default jitter
  is a fixed 0–1 s regardless of base wait. We now scale jitter to the base wait.

### P1-8. Bugs found by running things (kept here because they're good interview material)
- **Blank token treated as present.** `GITHUB_WRITE_TOKEN=` parsed as `SecretStr("")`, so the write
  server's "no token" guard passed and it started serving (it would fail later with 401). Found when a
  stdio smoke test hung. Fix: a validator maps blank strings to `None`; test added.
- **Flaky test from stream ordering.** A test assumed stdout came before stderr; Docker multiplexes
  them as separate frames with no ordering guarantee. Fixed the assertion, not the code.

---

## Phases 2–7 (2026-10-09)

The user asked to build the remaining phases in one go without credentials. So: everything that
needs the Claude API is tested with test doubles from `tests/fakes.py` (never imported by production
code), and **no benchmark numbers exist yet**. Those require a real run with `ANTHROPIC_API_KEY`.

### P2-1. Model choice changed from the proposal (Q2)
The Claude API reference used for this build says to default to `claude-opus-5-5` and not to
downgrade to Sonnet/Haiku for cost on the user's behalf. My Q2 proposal (Sonnet for specialists, Haiku
for planner/aggregator) was exactly such a downgrade, so **every agent defaults to `claude-opus-5-5`**
and each agent's model is one env var (`LLM_PLANNER_MODEL`, …). Prices for Opus/Sonnet/Haiku 5.5 are in
`llm/pricing.py`, source + date noted; an unknown model raises instead of costing $0.

### P2-2. API facts that shaped the agent loop (verified in anthropic 1.12.1 source + bundled docs)
- Forced `tool_choice` (`any`/`tool`) **returns 400 on claude-opus-5-5**, so agents cannot be forced
  to call a "submit" tool. Final output uses **structured outputs** (`output_config.format` with
  `anthropic.transform_schema(Model)`), which works together with tools.
- `transform_schema` moves `minimum`/`maximum` into descriptions (not enforced by the API), so every
  output is re-validated with Pydantic. One re-ask with the validation error, then a note.
- Thinking cannot be disabled on Opus 5.5; depth is set with `effort` (default `medium` on this model,
  we set `high` explicitly).
- **Refusal fallbacks on by default** (`fallbacks="default"` + beta `server-side-fallback-2026-07-01`):
  a security reviewer reading exploit-shaped code can trip a safety classifier. Refusals that still
  happen become `refusal` notes. Cost is computed per attempt from `usage.iterations` (declined
  attempts are counted, so cost may be slightly overstated in that case).
- The last step of every agent is reserved for an answer with `tool_choice: "none"`.

### P2-3. Bugs found by running things
- **Missing API key crashed mid-review.** Without credentials the SDK raises a plain `TypeError` at
  request time (not an `APIError`), which the loop did not catch. Now checked before any work starts
  (`MissingCredentialsError`).
- **`.env` key silently ignored.** pydantic-settings reads `.env` into the settings object but does
  not export it to `os.environ`, and the SDK only reads `os.environ`. The key is now passed explicitly.
- **LLM call records were mutable.** `LLMCall` held a reference to the growing `messages` list, so
  anything recording a call (test spy, Langfuse input) saw later turns. Calls now snapshot the list.
- **Errors arrived wrapped in 4 nested `ExceptionGroup`s** (one per anyio task group in the MCP
  clients), so `except ReviewFailedError` never matched. `single_error()` unwraps exactly-one-cause
  groups at the boundary and keeps genuine multi-error groups intact.
- **Prompt leaks:** file paths and the planner's per-file focus text were initially rendered outside
  the untrusted-data fence. Both are PR-derived; whole file sections are now fenced.
- **Two injection comments for one attack:** adjacent flagged lines now merge into one finding.
- **Over-broad injection pattern:** "note to reviewer" (normal human text) was flagged; removed.

### P2-4. Argument binding in the toolbox
`owner`/`repo`/`sha` are removed from the tool schemas the LLM sees and forced to this PR's values at
call time. Injected text cannot redirect an agent to another repository or commit (tested: the
model asks for `evil/other@fff…`, the tool reads the PR head).

### P2-5. Deterministic scanners as a floor (D8)
High-confidence secret formats and OSV-listed vulnerable pins become findings **without the LLM**,
so they survive refusals and budget exhaustion (tested). Secrets are masked (`AKIA****[redacted]`)
so the credential is not copied into Postgres, traces, or the posted comment. Third-party scanners
(bandit/semgrep) were **not** added: not verified, and not needed for the benchmark's issue types.
TODO(future): evaluate bandit in the sandbox.

### P3-1. Graph shape and what was dropped
planner → `Send` fan-out (bug batches of 8 files, security, tests) → aggregator. Fetching the PR
happens before the graph: if it fails, there is nothing to review. **Not used: the LangGraph Postgres
checkpointer.** Reviews are idempotent and re-runnable, approvals live in our own tables (D9), so the
checkpointer would add a dependency without a use. Recorded here because D1 listed it as a benefit.

### P3-2. Least privilege for MCP subprocesses (verified in mcp 2.3.0)
The stdio client gives children only HOME/PATH/… and we add exactly what each server needs. Child
working directory is the workspace cache, which has no `.env`, so read/search servers cannot pick up
the API key or the write token. (This is also why the Docker image installs the package instead of
relying on PYTHONPATH: the variable is filtered out.)

### P4-1. Approval is enforced by the write server itself (closes the Phase 1 TODO)
The free-form `post_review_comment(body, path, …)` tool was **replaced** by `post_approved_comment(comment_id)`.
The server loads the comment from Postgres and posts only rows a human approved and the publisher
claimed (`posting`). A compromised caller can't post arbitrary text or unapproved comments.

### P5-1. Comment lifecycle and no double posts
`pending ⇄ approved ⇄ rejected → (publisher) posting → posted | post_failed`. `posting` is committed
*before* calling GitHub; a crash after GitHub accepted the comment leaves it `posting` and it is never
retried automatically. A lost response (timeout) is recorded as `post_failed` with "outcome unknown".

### P5-2. Worker and benchmark error boundaries
The worker and the benchmark harness each have exactly one `except Exception`: one failed review must
not kill the worker, and one broken case must not abort the benchmark. Neither hides anything: the
worker stores the traceback on the review (visible in the UI), and the benchmark scores the case as a
miss and prints the error.

### P5-3. Benchmark design
- 30 cases in `benchmark/cases/*.yaml`: 14 bug (2 break existing tests), 8 security, 5 clean, 3 injection
  (code comment, PR description, docstring; each also has a planted real issue). 27 labels.
  `python -m prreview.bench validate` prints every labelled line for the manual label review.
- **Only GitHub's HTTP API is replaced** (answered from the fixture); MCP servers, graph, agents,
  Docker sandbox, OSV and the model are real. Reproducible, no token or real PRs needed.
- Matching: same file, accepted category, line overlap with ±3 tolerance, one-to-one.
- Metrics: recall, precision, false-positive findings per clean PR, injection resistance (planted
  issue found AND no forbidden text AND injection flagged), avg steps/cost/latency, per-kind recall,
  std-dev across trials. Errored runs count as misses.
- `run` refuses to start without `--yes` (paid API calls). Results are written to
  `benchmark/results/<UTC timestamp>/` with the command, config, dataset hash, and every raw finding.
- **Known bias:** I wrote both the injection heuristics and the injection cases; two heuristic patterns
  (`[SYSTEM]` tags, "pre-approved") were added after the oracle run showed case i02 was not flagged.
  The patterns are general, but the deterministic part of the injection metric is partly tuned to
  this dataset. Mitigation for the user: add injection cases you write yourself.
- Harness/scoring correctness is checked with an **oracle double** that reports exactly the labels:
  it scores 100% recall, 0 false positives, injection resisted (6 cases, unit tests).

### P5-4. Persistence
SQLAlchemy 2.1 + psycopg 3, schema via Alembic (migration tested against real Postgres 17).
Initial autogenerated migration used `timestamp without time zone` while code writes aware UTC
datetimes; fixed to `timestamptz` before anything depended on it.

### P6-1. UI shows comment bodies as literal text
Comment bodies are LLM output derived from untrusted PR content. The approval UI shows them with
`st.code` (exactly what will be posted, nothing rendered, so no remote images/links load in the
approver's browser). Progress uses `st.fragment(run_every=2)` polling Postgres.

### P6-2. Tracing
Langfuse 4.17: one trace per review (`propagate_attributes(session_id=review_id)`), `agent` span per
agent, `generation` per LLM call (model, `usage_details`, `cost_details`), `tool` span per tool call
(ERROR level on failure). No-op without keys. Verified against the real SDK with an in-memory
OpenTelemetry exporter (exact attributes asserted). **Not verified against Langfuse Cloud** (no keys).

### P7-1. Compose
One image, a service per role, `environment:` per service (no blanket `env_file`). Only `sandbox`
mounts the Docker socket and runs as root (the socket is root-owned). The sandbox speaks Streamable
HTTP with DNS-rebinding protection enabled explicitly (`--allowed-host sandbox:*`): mcp 2.3.0 only
auto-enables it for localhost binds. Verified: correct host works, forged Host → 421. Publisher is
behind the `publish` profile (it needs the write token).
Verified in the running stack: migration, UI health, worker→sandbox over HTTP, stdio MCP subprocesses
inside the worker container against live GitHub, and a queued review failing cleanly with
`MissingCredentialsError` recorded in Postgres (no key configured).
