# Project Agent Workflow

These instructions apply to any AI assistant working in this repository.
Apply them within higher-priority platform, tool and safety requirements; do
not promise capabilities or indefinite turn lifetime that the environment lacks.

**Active continuation rule:** Once the user starts or resumes open-ended project
work, keep executing useful authorized work until the user explicitly stops,
pauses or ends that work. Checkpoints, summaries and commits are progress, not
completion. Apply the final-answer gate in Continuation and Stopping below.

**Standing authority:** Independently scope and complete reversible local
implementation, compatible design choices, bounded synthetic unit fixtures,
required tests, documentation, checkpoint commits and periodic normal upstream
pushes within the agreed plan. Choose recommended defaults and record rationale;
do not ask for routine planning/testing approval or a finish/continue vote.
This supersedes historical routine approval wording, not protected limits,
exhausted experiments or missing qualification evidence. Fresh bounded local
source-recovery and unit/regression checks are delegated even when historical
routine test accounting is missing; preserve unknown attempts, not approval gates.

## Project

`neural-compiler` investigates **LLM-as-compiler with a native CPU execution
bridge**. The model translates intent into a compact program; the host bridge
validates, compiles where needed, executes and returns results into inference.
Python with standard-library intrinsics and arbitrary-precision integers is the
default path. C is a compute-tier candidate; typed DSL and constrained WAT
emission are research tracks, not implemented production compilers.

The current target is `Qwen/Qwen3.5-4B`, a hybrid full-attention/Gated DeltaNet
model. Preserve the same live KV and recurrent cache across tool calls, or use
verified exact-boundary hybrid prefix-cache checkpoints. Native tool framing
comes from the tokenizer/chat template, not handwritten turn boundaries.

Current source/raw-output budgets are 64 KiB; native call bodies default to
4,096 tokens, response turns to 2,048 tokens, and the native experiment to five
serial calls. Verify the owning implementation before use. These defaults are
not approval for a model run or evidence of resource fit.

This is a **P0 prototype, not a production security boundary**. Supervised
per-request execution, in-jail C/assembly compilation, isolated Wasm/unicorn
harnesses, native protocol framing and single-sequence cache continuation exist.
Opt-in seccomp plumbing exists, but a production syscall policy, per-request
cgroups/PID caps, a curated-import policy and production serving integration
remain unfinished. Do not expose it to untrusted workloads.
The numerical NeuroVM helpers and Part II of [WHITEPAPER.md][whitepaper] are
retained research/reference material, not the active CPU-bridge execution path.

## Current Work

Read [PROJECT_DIRECTION.md][direction], especially its implementation audit and
P0 continuation, for the current maturity, evidence limits and next gates.
Its architecture and roadmap are targets; the owning code and tests determine
what is actually implemented. Historical E1-E3b results are prototype evidence,
not new native-protocol measurements or renewed experiment authority.

Continue this queue, deriving smaller compatible prerequisites when a gate is
blocked:

1. Build bounded offline prerequisites for randomized held-out-family evaluation:
   task/checker contracts, synthetic unit fixtures and no-tool/Python-tool/current
   best-tier harness paths now exist; advance matched-family reporting and checks
   using them. Implementing a harness does not authorize accepting
   a dataset, collecting trajectories or running a pretrained-model study.
2. Advance the P0 serving comparison: live pause/resume versus hybrid
   prefix-cached resubmission. Process-local CPU exact-boundary snapshots now
   exist; real-engine checkpoint integration remains outstanding.
   Verify cache/turn correctness before claiming
   latency gains; real-engine runs and dependency changes remain separately scoped.
3. Advance P1 containment code and tests: seccomp, per-request cgroups/PID caps,
   import policy, deadlines, compile-in-jail and independent containment evidence.
   Do not change host privileges/security or launch an untrusted service as a
   testing workaround.

SFT, execution-rewarded RL, middle-level IR investment and production serving
remain downstream of the relevant P0/P1 gates. Fine-tuning is part of the project
direction, not blanket permission for arbitrary training/data/resource runs.
Detailed milestones, failed checks, measured results and handoffs belong in
dated updates to [PROJECT_DIRECTION.md][direction], not here. Replace this short
state as it changes; do not append a session-history catalogue.

## Start With Context

- Read Current Work and the newest relevant implementation/experiment update in
  [PROJECT_DIRECTION.md][direction] before continuing implementation. Use its
  decision, architecture, training and roadmap sections as the reading map.
  Read [WHITEPAPER.md][whitepaper] for the relevant mathematical/reference scope,
  not as an instruction to rebuild an in-weights VM. Do not load entire archives
  for routine work.
- Check the actual worktree, nearby implementation/tests and applicable local
  instructions. Preserve user changes, including editor settings and untracked
  files. After an interrupted tool call, verify its outcome before repeating it.
- Verify selected environments, cached files, pinned revisions, available tools
  and resource budgets rather than assuming they survived a restart. Reuse
  existing caches and environments; do not duplicate downloads or checkouts.
- Distinguish proposed, approved, implemented, tested, integrated and measured
  work. A historical approval or completed bounded run is not renewed permission.
  Apply the current standing authority to new routine local work; do not treat
  older per-checkpoint approval wording as a reason to ask again or stop.

## Documentation Audience

- [README.md][readme] is the public introduction: capabilities, maturity limits,
  usage, documentation links and licensing. Do not add routine progress updates,
  test counts, commit histories or session handovers. This rule does not request
  a separate cleanup of existing documentation.
- Keep architecture decisions, accepted requirements, contracts and rationale
  in [PROJECT_DIRECTION.md][direction]; keep mathematical/reference explanations
  in [WHITEPAPER.md][whitepaper]. Keep changing status, commands/results,
  failed-check history, evidence limits and handovers in concise dated updates
  to the existing direction/review documentation. Preserve historical reviews
  and recorded [experiment results][experiments] as historical evidence.
- Keep reading maps short; archive whole older sections before they become
  unwieldy, roughly 1,000 lines for a reading map and 2,500 for a topic/archive.
  Rebase local links and preserve historical anchors, failures, approvals, spent
  ledgers and evidence limits. These are organization targets, not experiment
  budgets or permission to reorganize unrelated documents.
- Use [.devcontainer/README.md][container-guide] for reusable environment/setup
  instructions. Use existing documents rather than creating new Markdown files
  unless asked. Use reference-style links and check their targets.
- Keep stable workflow rules here. Replace Current Work and task summaries as
  they change rather than appending more historical checkpoints to this file.

## Work Independently

- Do not invoke agent-controlled questionnaire/choice dialogs. Do not spawn
  subagents unless explicitly requested.
- Own planning and execution: choose the first useful unblocked task, record its
  finite scope/resource budget and exit condition, implement, verify, checkpoint,
  then start the next useful authorized action in the same turn. Missing routine
  planning/testing approval is not a blocker.
- Assistant-written proposals and test plans are not permission gates for work
  already delegated. Do not replace implementation with repeated documentation-
  only budget proposals, approval requests or checkpoint reports. Record the
  bounded routine plan and execute it within the same work slice.
- At material checkpoints or new evidence, reassess the plan and priority queue;
  update compatible choices independently. Apply delivered user corrections
  promptly. Reviews are not new permission gates or reasons for repeated audits.
- Form a falsifiable local hypothesis from the owning code and a nearby test.
  Make the smallest useful edit and immediately run a focused check before
  expanding scope. Reuse existing tests/helpers and avoid unrelated cleanup.
- For an isolated blocker, name the missing evidence, authority, tool or input,
  retain the conservative state, prepare useful prerequisites and continue other
  unblocked work. Diagnose tool failures and use supported alternatives.
- Give concise progress updates: what changed, what was verified and what
  remains. Do not manufacture work merely to keep a turn alive.

## Continuation and Stopping

- Treat an open-ended continuation as active across checkpoints, commits,
  pushes, progress reports, tool notifications, reloads and context compaction.
  Completing one task does not complete the session or require another continue.
- Do not end active work with a routine permission/finish vote. An assistant-
  imposed planning gate or missing routine test ledger is not an all-path blocker;
  use the recovery rules below and take the next useful authorized action.
- During open-ended work, process new user messages delivered by the platform
  before the next slice; apply the newest compatible instructions. Do not claim
  inbox polling or message checks with unavailable tools.
- Determine current user intent from actual user instructions, not an assistant
  recap, generated conversation summary, quoted transcript or historical stop.
  These are context, not new user commands. A recap describing an old pause or
  summary-only request cannot end a subsequently resumed continuation. If the
  latest actual request is unavailable, recover it from available session
  history before deciding to stop; do not invent a stop instruction.
- An explicit current stop, pause or finish request takes priority. Stop safely
  without confirmation; cancel or contain owned work, retain evidence and leave
  a concise handoff. Do not start another slice or delay for unrelated checks,
  commits/pushes or a finish vote. Quoted stop words or a request to stop
  interrupting are not requests to pause project work.
- Stop/finish instructions apply to the current work, not future continued
  sessions. A fresh continue or changed task requires evaluating the latest
  request; do not carry an old finish permission across checkpoints, later turns
  or context loss. After a pause, a status question or terminal notification is
  not permission to resume implementation.
- Respect explicit task boundaries. A standalone documentation-only request,
  review or proposal is not permission to start separate implementation or
  collect data. During an already active continuation, handle a side request
  within its scope, then resume the existing authorized queue unless the user
  explicitly pauses, replaces or limits that continuation. An explicit "only
  update documentation", "review only" or equivalent limit must be honored;
  standing authority never expands that limit or authorizes protected actions.
- Use commentary for checkpoint reports and side-request results during active
  continuation, then execute the next concrete action. Do not issue a checkpoint
  final answer, offer to continue or leave an authorized next step for another turn.
- Use event-based waiting only when a supported tool is actually waiting for
  authorized work or required input; follow its notification protocol. Do not
  use sleep loops, repeated polling, dummy jobs or invented work to keep a turn
  alive. Instructions alone cannot provide background message monitoring.
- If every useful in-scope path is genuinely blocked, or platform/tool limits
  prevent execution, report the exact constraint and pending action; do not
  pretend to remain active. Missing qualification evidence, an exhausted
  experiment or an isolated blocker does not end compatible prerequisite work.
  Do not evade protected limits or invent work to avoid stopping.

**Final-answer gate.** Before ending a turn, verify one of these applies:

- The user explicitly stopped, paused or ended the active work.
- A standalone bounded request is complete, with no active continuation to resume,
  or the user explicitly replaced the continuation with that bounded request.
- Every useful authorized path is genuinely blocked after checking compatible
  prerequisites and the remaining agreed queue; identify the precise blockers.
- A higher-priority safety requirement or an actual platform/tool limit prevents
  further work; state the concrete constraint without claiming to remain active.

Otherwise, the next action must be useful authorized work, not a final answer.
A clean worktree, green gate, checkpoint, commit, push, recap or completed side
request is not a stopping reason. This is an instruction-level behavior rule,
not a guarantee of unlimited execution or background message monitoring.

## Verification and Commits

- Prefer bounded synthetic fixtures for routine arithmetic, protocol, cache and
  executor checks. Default tests need neither downloads nor a GPU. Tiny untrained
  CPU model forwards are unit fixtures, not pretrained inference studies.
  Cached-tokenizer integration skips when absent; unavailable bubblewrap
  namespaces are reported as skips/errors, never an unjailed execution fallback.
- Required focused/regression checks for meaningful source changes are delegated.
  Record finite fixture/query counts, attempt/resource caps and failure/cleanup
  plans before execution. Count failures/aborts; retain tolerances, hard work
  limits and per-family controls.
- If historical routine test accounting is missing, mark interrupted/unknown
  attempts as spent/unknown and start a fresh bounded revalidation ledger under
  standing authority. Run the necessary focused and root checks without another
  approval request; never infer a past pass, refund old attempts or treat a new
  ledger as qualification evidence. This does not reopen protected experiments
  or override an explicit user-imposed execution ceiling.
- Distinguish explicit user-imposed hard caps and protected experiment budgets
  from assistant-selected routine test plans. For the latter, diagnose a failure
  before any rerun and document a narrowly revised finite plan when a necessary
  repair or required verification needs it. Do not repeatedly reset budgets for
  unchanged failures, rerun unchanged successful gates, relax fixture/work/safety
  limits or manufacture a new scope to evade a protected cap.
- Old spent experiment ledgers remain historical and are not refunded. Existing
  fixtures may run as unchanged controls in the required regression gate for a
  new source change; this does not authorize standalone old experiments, extra
  study runs or deferred audits. A substantive repair permits another necessary
  check within the recorded scope. Increasing an explicit user-imposed hard cap
  or protected experiment budget still needs explicit authority.
- After a substantive edit, run the cheapest behavior-scoped check first. Before
  each source checkpoint commit, run focused checks for the changed behavior and
  the default gate from the repository root:

```sh
python -m pytest -q && python -m ruff check . && git diff --check
```

- Run `python -m ruff format --check <changed Python files>` as well. Leave
  unrelated formatting alone, including `.devcontainer/smoke.py`.
- For native/runtime changes, ensure relevant jailed Python/C/assembly and
  Wasm/unicorn checks actually ran. Use strict C99 and compiler/sanitizer controls
  when the changed behavior needs them; toolchain availability is not proof of
  compile-in-jail, containment or sanitizer coverage. Report skips and missing
  prerequisites accurately.
- For protocol/cache changes, verify exact turn/token accounting, pending-token
  handling and multi-call logits against full-prefill reference behavior. Tiny
  CPU Torch-reference tests are not FLA/Triton, 4B accuracy, GPU fit or serving
  latency evidence. Run numerical audits only for the relevant changed claims.
- Do not relax tolerances, assertions, deadlines, token/tensor bounds or safety
  checks for green results. Diagnose before rerunning; substantive repairs permit
  necessary checks within scope. Do not invent inherited xfails or carry over
  unrelated projects' test exceptions.
- Do not rerun an unchanged successful gate merely for reassurance. For
  documentation/instruction-only changes, validate scoped content, references,
  local targets and whitespace; state that the code suite was not rerun.
- Scoped checkpoint commits and periodic normal pushes are authorized. Commit
  meaningful verified increments. Review the diff and stage only checkpoint
  files; never include or discard unrelated user work.
- Keep the remote reasonably synchronized by pushing verified checkpoints
  periodically through the current branch's configured upstream, without an
  approval prompt. Check the worktree, branch and tracking state first. If the
  remote advanced, fetch and reconcile conservatively without discarding either
  history; validate affected work before a normal push. Do not force-push,
  silently change remotes or alter credentials/security settings to make it work.
- Verify the actual remote revision after a push; report remaining local changes
  explicitly. A pushed earlier checkpoint does not include later uncommitted work.
- If synchronization is blocked, retain the local commits, record the precise
  cause/pending action and continue other useful work. Diagnose before retrying;
  do not invent credentials or repeatedly retry an unchanged failed push.
- Use Conventional Commit subjects (`type(scope): imperative summary`, with scope
  optional) and a real bullet body. Branch creation, force-pushes and history
  rewriting still need explicit direction. Never revert another person's changes.

## Protected Boundaries

Where fresh explicit authorization is absent, choose not to take the protected
action; record the reason and recommended next action without an approval
questionnaire, and continue other useful in-scope work:

- Changes to accepted project requirements, metric definitions, safety thresholds
  or frozen protocol/execution contracts. A compatible implementation choice is
  delegated; a new requirement is not silently adopted as a convenient default.
- New model-study runs, task labels/imports, dataset/cohort/split acceptance,
  teacher labeling, trajectory collection, SFT/RL, outcome-driven tuning or
  benchmark-policy adoption outside current authorization. Exhausted protected
  run approvals stay exhausted; synthetic unit fixtures are not accepted task data.
- Running third-party model code at an unapproved revision, using
  `trust_remote_code=True`, or starting experiments beyond the exact current
  runtime/resource/run scope; new or upgraded dependencies and `.devcontainer`
  changes outside the agreed scope.
- Downloads other than approved revisions/artifacts, and any download that would
  leave less than about 20 GB of free disk space. Existing local cache inspection
  and bounded offline unit fixtures do not authorize new downloads.
- Deleting files/data you did not create, force-pushing/rewriting git history,
  creating branches/changing remotes, external account setup, payments, rented
  compute, licences or contacting people. Normal checkpoint pushes through the
  existing configured upstream are covered by standing authority.
- Untrusted workload exposure, privileged or shared-host changes, security/TLS
  bypasses and hardware access outside the agreed scope. Never disable
  namespaces, resource limits or validation to make an experiment run.

Never convert autonomy into evidence: mathematical correctness, containment,
task quality and latency/fit still require actual acceptance evidence. Hashes,
passing fixtures and agent defaults are not substitutes. Prepare authorized
prerequisites and record the precise protected action, expected evidence,
resource budget and cleanup plan for a future user directive. Never request
passwords, tokens or other secrets. Do not reopen settled choices without new
evidence or a changed claim/domain. After three failed approaches to the same
problem, retain the failures, mark the blocker and switch to other useful
in-scope work rather than repeating the same attempt.

## Rules

- Evidence: cite primary sources and label estimates as estimates. Never present
  unmeasured latency, GPU fit, accuracy or containment as fact. Compare targets
  with matched tasks, intrinsics, prompts, repair budgets, hardware and serving
  settings; include no-tool and Python-tool baselines. The primary objective is
  wall time/GPU-seconds per correct answer with uncertainty, not token count alone.
- Third-party models/data: pin/review revisions, verify manifests, licences,
  selected environments and current exact run authority. Historical approvals
  do not renew runs. Record evidence/authority in [PROJECT_DIRECTION.md][direction].
  Do not transfer another project's teacher/model licences or commercial-use
  assumptions into this repository.
- Serialize GPU jobs. The historical audit host was an RTX A3000 12 GB, not a
  guarantee of current hardware or capacity. Verify CUDA/BF16 support, available
  memory, disk headroom and finite token/time/memory budgets before each authorized
  job; retain stricter run-specific caps. Do not inherit another model's budgets.
- Dependencies: without explicit authorization, keep the current environment
  and use installed tools. For an authorized change, keep [pyproject.toml][package]
  and applicable [CPU][cpu-input]/[CUDA][cuda-input] inputs consistent, then
  regenerate applicable Python 3.12 CPU/CUDA and Python 3.10 compatibility hash
  locks with [the documented commands][container-guide]. Never disable TLS
  verification or `--require-hashes`.
- Keep weights, datasets, caches, private traces and large generated artifacts
  out of git; use ignored directories and add scoped ignore rules when needed.
  Preserve checked-in historical benchmark JSON unless explicitly regenerating
  that experiment. New native results are separate, not replacements for E3.
- Execution: the host bridge is the security boundary. Compile generated native
  code inside the jail; keep Wasm/unicorn out of the supervisor process. Guest
  stdout is data, never a trusted status/control channel. Preserve explicit
  compile/runtime/protocol/timeout/output-limit/isolation failures.
- Protocol/cache: derive and validate delimiter IDs/chat-template turns; escape
  guest control markers, enforce call/result budgets with visible truncation,
  and preserve each sampled token exactly once. Do not arbitrarily slice or
  roll back Gated DeltaNet recurrent state like ordinary attention KV.
- Put product Python code in [src/neural_compiler/][source], reusable bridge logic
  in [bridge/][bridge], and focused regressions in [tests/][tests]. Keep research
  scripts and historical results in [docs/review/experiments/][experiments].
  Reuse local conventions/helpers; do not recreate the copied project's layout.
- Environment/setup: [.devcontainer/README.md][container-guide]. The configured
  default is Python 3.12 at `/opt/venv`, with Python 3.10 compatibility CI, not an
  assumption that either interpreter survived a restart. Model/data/training
  and hardware commands are opt-in under current protected authority.

## Leave a Clear Handoff

- At meaningful checkpoints, update [PROJECT_DIRECTION.md][direction] with the
  actual change, commands/results, failed-check history, evidence limits,
  artifact retention/cleanup and next concrete action. Keep accepted architecture
  decisions distinct from dated results and the concise Current Work here.
- Keep raw logs for long-running authorized jobs in ignored directories. Run
  large-memory/GPU jobs serially and report measured resource use without
  presenting process counters as hardware guarantees.
- After a restart, verify file sizes/tails and live processes before repeating
  work. Preserve damaged and uncommitted versions before recovery; use bounded
  reads and summaries instead of whole-file diffs or historical log dumps.
  Never clear user/editor caches or attribute a crash to OOM without evidence.
- Treat the host as potentially shared. Contain and reap only owned jobs; do not
  kill unrelated processes or make broad cleanup/security changes.
- When the final-answer gate permits a handoff, include commit IDs, verification,
  remaining boundaries and worktree state. Never continue after a stop or pause.

[direction]: docs/review/PROJECT_DIRECTION.md
[whitepaper]: WHITEPAPER.md
[readme]: README.md
[container-guide]: .devcontainer/README.md
[package]: pyproject.toml
[cpu-input]: .devcontainer/requirements.in
[cuda-input]: .devcontainer/gpu/requirements.in
[source]: src/neural_compiler/
[bridge]: src/neural_compiler/bridge/
[tests]: tests/
[experiments]: docs/review/experiments/
