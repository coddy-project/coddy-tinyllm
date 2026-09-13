# 1. The problem

## One sentence

Given a Coddy config file and a free-form instruction in Russian or English, decide whether the
instruction touches the configuration at all, and if it does, emit the `config_set` command list
that carries it out - locally, on a CPU, in under a second, without calling a cloud model.

## Why this is worth building

Coddy already has the whole machinery for a machine to edit its own config. `internal/tools`
carries `config_get`, `config_set`, `config_changes`, `config_commit`, `config_revert`,
`config_rollback` and a staging area, and `internal/config/uci.go` defines a four-verb command
language with a parser and a dry-run:

```
set      <path>=<value>
add_list <path>=<value>
del_list <path>=<value>
delete   <path>
```

Paths are dotted, with `name[key=value]` selectors for list entries (`mcp_servers[name=context7].command`).
Nothing is applied until `config_commit`, and a commit keeps a snapshot that `config_rollback`
restores.

What is missing is the cheap half. Today "turn off the Telegram gateway" costs a full round trip to
a frontier model, with the whole tool catalogue and system prompt in the prefix, and it does not
work at all when the operator has no key, no network, or is on the free tier of a provider that is
currently rate-limiting them. The task itself is narrow: a few hundred addressable fields, four
verbs, values that are mostly booleans, integers and short enums.

That is exactly the shape of problem a task-specific small model is good at, and exactly the shape
of problem a frontier model is a waste on.

## The contract

The component is a **fast path in front of an existing, already-safe tool**, never a replacement
for the agent.

```
instruction + config + schema
        |
        v
  [ gate ] --------- not a config request ---> return NOOP, agent continues as usual
        |
        v
  [ retrieve ] -> candidate fields from the live schema
        |
        v
  [ generate ] -> UCI command lines, grammar-constrained
        |
        v
  [ verify ] -> ParseUCICommands + DryRunUCICommands + schema check
        |            |
        |            +--- fails twice ---> escalate to the normal LLM path
        v
  staged edits, shown to the operator, committed only by config_commit
```

Three properties fall out of that and they are what make the project shippable:

- it is allowed to be wrong, it is not allowed to be silently wrong - every output passes the same
  parser and dry-run the big model's output passes, and the operator still confirms the commit;
- it is allowed to give up - an abstention costs one escalation to the path that exists today;
- it never invents a field name - the decoder is constrained to paths that exist in the schema
  loaded at runtime.

## The final case, stated precisely

Input:

- `config.yaml` as it is on disk (a real one is ~900 tokens; power users reach several thousand);
- one instruction, Russian or English, one to three sentences;
- `config.schema.json`, which is already embedded in the Coddy binary and changes with every release.

Output, one of:

- `NOOP` with a reason class (not about settings / already in that state / needs a human);
- an ordered list of UCI commands that parse, dry-run clean and validate against the schema.

Hard requirement: **the schema changes without retraining the model.** A new Coddy release adds
`swarm.rings[].fanout`, and the same model file must handle a request about it on day one.

## Non-goals

- Not a chat model. One turn in, one command list out, no dialogue, no explanations.
- Not a code editor and not a shell. "Primitive commands" here means the config verbs, plus
  possibly a small closed set of local actions later (see [07-roadmap](07-roadmap.md)); it does not
  mean free-form `run_command`.
- No multi-hop reasoning, no planning, no tool loops.
- Not a replacement for the agent when the request is ambiguous - ambiguity is an abstention.

## Success criteria

Measured on a held-out set built the way [06-eval](06-eval.md) describes:

| Metric | Floor to ship | Target |
|---|---|---|
| **False-edit rate** `P(EDIT \| turn is not about config)`, explicit mode | 1e-2 | 1e-3 |
| **False-edit rate**, ambient mode | 1e-4 | 1e-5 |
| Missed-request rate `P(NOT_CONFIG \| turn is a config request)` | 0.10 | 0.03 |
| Exact command-set match on config requests | 0.70 | 0.85 |
| **Valid-but-wrong rate** (parses, dry-runs, validates, wrong field or value) | 0.05 | 0.01 |
| Schema-valid output when it does emit commands | 0.99 | 1.0 |
| Retrieval recall@12 on fields added after the training cut | 0.95 | 0.99 |
| Unseen-field generalisation (fields added after the training cut) | within 10 pts of seen fields | within 5 |
| Latency, gate only, 4 CPU threads | 150 ms | 50 ms |
| Latency, full generation, 4 CPU threads | 1.5 s | 500 ms |
| - of which prefill of the retrieved prompt | **unmeasured, plausibly 1-5 s** - see [03-architecture](03-architecture.md) §3.5 | |
| Artefact size added to the binary or `~/.coddy` | 400 MB | 150 MB |

The first row is the one that decides whether this can run on every turn or only behind an explicit
command, and it is deliberately stated as a **rate of the dangerous event** rather than as a
precision or recall figure. Those are easy to state backwards: if NOOP is the positive class, an
unwanted edit on a non-config turn is a NOOP *false negative*, so it damages recall, and a
comfortable-looking "recall floor 0.90" would license editing the config on one in ten unrelated
turns. Name the hazard directly instead.

The latency rows are a budget, not a measurement, and the prefill line is currently a **known
threat to the budget rather than a commitment**: a ~1 000-token prompt through a 0.6 B decoder on four
CPU threads is plausibly seconds, which is why M3 measures it before anything is trained and why
[03-architecture](03-architecture.md) §3.7 exists.

The ambient target has a sample-size consequence worth knowing before promising it: demonstrating a
false-edit rate of 1e-4 with zero observed failures needs roughly **30 000 representative negative
turns** for a 95 % upper bound (rule of three, 3/n). That is the real cost of shipping ambient mode,
and it is a data-collection cost, not a modelling one.

## The floor this has to beat

Before any of it is justified, the same eval set gets run against a **no-neural-network baseline**:
BM25 over the field cards, plus a hand-written rule set for the twenty most common intents
(enable/disable X, set N, change the model, add an MCP server). If that scores within a few points
of the model on the real request distribution, the model is 100 MB of dead weight and the project
should stop at the retriever. This is written down first on purpose - see
[07-roadmap](07-roadmap.md), milestone 0.
