# Agent Preferences

## Testing

- Use **subtests** (`t.Run`) to group related test cases under a single test function.
- Use **[testify](https://github.com/stretchr/testify)** for assertions:
  - `require` for conditions that must pass for the test to continue (fatal on failure).
  - `assert` for conditions that should be checked but don't need to halt the test.
- Use `assert.InDelta` for floating-point comparisons instead of exact equality.
- Use `assert.ErrorIs` to check sentinel errors.
- Prefer table-driven tests when there are many similar cases; use subtests for distinct scenarios.

## Go Style

- Keep packages small and focused.
- Use `internal/` for packages that should not be exposed outside the module.
- Use type aliases in `internal/types` to wrap third-party types (e.g., gonum).
- Define sentinel errors (e.g., `var ErrDimensionMismatch = errors.New(...)`) for reusable error conditions.
- Wrap sentinel errors with `fmt.Errorf("%w: ...", err)` to add context.

## Tooling

- Linter: `golangci-lint` v2 with config in `.golangci.yml`.
- Run `make audit` to lint the project.
