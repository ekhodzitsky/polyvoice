# Agent Flow — polyvoice

This is the default way an agent works inside this COAD-native repository.

COAD repository: https://github.com/ekhodzitsky/coad

## Entry Protocol

1. Read root `AGENTS.md`.
2. Locate the relevant module contract.
3. Read the module `README.md`.
4. Read the module `TODO.md`.
5. Identify surfaces, consumers, invariants, and proof commands.
6. Identify whether the workcell is leaf or composite.
7. Define read scope and write scope before editing.
8. Acquire or declare the write lease if implementation changes are needed.
9. Inspect implementation only after the boundary is clear.

## Two-Minute Orientation Checklist

Before editing, the agent should be able to answer:

- Which module owns this behavior?
- Is the owning workcell leaf or composite?
- Which files am I allowed to change?
- Which files may I read for context?
- Which surface am I changing or preserving?
- Which consumers could be affected?
- Which invariants must remain true?
- Which proof commands will establish correctness?
- Do I have the only active write lease for this workcell?
- Does this require updating a contract, README, TODO, schema, or docs?

## Work Protocol

1. Lock the intended write scope and write lease.
2. Read the focused implementation files.
3. Add or identify proof before changing shared behavior.
4. Make the smallest change that satisfies the task.
5. Update module context if ownership, surfaces, dependencies, consumers,
   invariants, or verification changed.
6. Run focused proof.
7. Run broader verification when the change touches shared behavior.
8. Run `coad check .` before claiming the repository still follows COAD.

## Handoff Template

```markdown
Module:
Workcell:
Write scope:
Write lease:
Changed files:
Contracts/context updated:
Surfaces changed:
Consumers affected:
Invariants checked:
Proof run:
Known gaps:
Recommended next step:
```

## Review Protocol

A reviewer checks the contract before the diff:

1. Does the change stay inside declared module ownership?
2. Did it alter public or internal surfaces?
3. Are consumers updated or explicitly unaffected?
4. Are invariants preserved?
5. Is proof strong enough for the change class?
6. Did module context change when the module changed?

## What Agents Should Avoid

- Starting with a whole-repo scan when a module contract exists.
- Editing across module boundaries without updating ownership context.
- Editing a leaf workcell that already has an active write agent.
- Letting a parent orchestrator directly change child implementation files.
- Changing public surfaces without naming affected consumers.
- Reporting completion without proof output.
- Leaving local TODO or README stale after changing module behavior.
- Treating `coad check .` as a substitute for understanding the module.
