# Claude Guidelines

## Critical Guidelines

1. **Always communicate in English** - Regardless of the language the user speaks, always respond in English. All code, comments, and documentation must be in English.

2. **Minimal documentation** - Only add comments/documentation when it simplifies understanding and isn't obvious from the code itself. Keep it strictly relevant and concise.

3. **Critical thinking** - Always critically evaluate user ideas. Users can make mistakes. Think first about whether the user's idea is good before implementing.

4. **Lint after changes** - After making code changes, always run `just lint` to verify code quality and fix any linter issues.

5. **No disabling linter rules** - Never use special disabling comments (like `# noqa`, `# type: ignore`, `# ruff: noqa`, etc.) to turn off linter rules without explicit permission. If you believe a rule should be disabled, ask first.
