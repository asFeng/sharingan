# Claude Module Doc: cli/

> AI-facing documentation for the CLI module

## Purpose
Command-line interface using Typer with Rich for pretty output.

## Current Stage: ✅ Complete

## Files

### main.py
**App**: `app = typer.Typer(name="sharingan")`

**Commands**:

1. `analyze` - Main analysis command
   ```bash
   sharingan analyze "prompt" --model MODEL -o output.html
   ```
   - Options: `--model/-m`, `--output/-o`, `--generate/-g`, `--max-tokens/-t`, `--layer/-l`, `--head/-h`, `--show/--no-show`, `--interactive/-i`

2. `dashboard` - Launch Gradio dashboard
   ```bash
   sharingan dashboard --port 7860 --share
   ```
   - Options: `--port/-p`, `--share/-s`

3. `info` - Model architecture info
   ```bash
   sharingan info Qwen/Qwen3-0.6B
   ```
   - Shows: layers, heads, GQA info, hidden size

4. `version` - Show version
   ```bash
   sharingan version
   ```

## Entry Points

```toml
# pyproject.toml
[project.scripts]
sharingan = "sharingan.cli.main:app"
```

```python
# __main__.py (for python -m sharingan)
from sharingan.cli.main import app
app()
```

## Rich Output

Uses Rich for styled terminal output:
- `SpinnerColumn` for progress
- `Table` for structured data
- `Panel` for boxed content

## Usage Examples

```bash
# Basic analysis
sharingan analyze "Hello world" -o output.html

# With generation
sharingan analyze "Once upon a time" -g -t 50 -o story.html

# Specific layer/head
sharingan analyze "Test" -l 5 -h 3 --interactive

# Launch dashboard
sharingan dashboard -p 8080

# Check model info
sharingan info Qwen/Qwen3-0.6B
```

## Potential Improvements

- [ ] Add `--device` and `--dtype` options
- [ ] Add `batch` command for multiple prompts
- [ ] Add `compare` command for model comparison
- [ ] Add `--quiet` mode for scripting
- [ ] Add `--json` output format option
- [ ] Add shell completion generation
