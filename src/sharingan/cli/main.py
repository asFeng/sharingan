"""CLI commands for Sharingan."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="sharingan",
    help="Attention visualization for transformers",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    prompt: Optional[str] = typer.Argument(None, help="Text to analyze (or use --file)"),
    model: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--model",
        "-m",
        help="HuggingFace model name or path",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Read prompt from text file",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (HTML or image)",
    ),
    generate: bool = typer.Option(
        False,
        "--generate",
        "-g",
        help="Generate new tokens",
    ),
    max_tokens: int = typer.Option(
        50,
        "--max-tokens",
        "-t",
        help="Maximum tokens to generate",
    ),
    layer: Optional[int] = typer.Option(
        None,
        "--layer",
        "-l",
        help="Specific layer to visualize",
    ),
    head: Optional[int] = typer.Option(
        None,
        "--head",
        help="Specific head to visualize",
    ),
    show: bool = typer.Option(
        True,
        "--show/--no-show",
        help="Display visualization",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Use interactive Plotly visualization",
    ),
):
    """Analyze attention patterns for a prompt."""
    from sharingan import Sharingan

    # Handle input from file or argument
    if file:
        if not file.exists():
            console.print(f"[red]Error:[/red] File not found: {file}")
            raise typer.Exit(1)
        prompt = file.read_text().strip()
        console.print(f"[dim]Loaded {len(prompt)} characters from {file}[/dim]")
    elif not prompt:
        console.print("[red]Error:[/red] Provide a prompt or use --file")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(style="red"),
        TextColumn("[bold red]{task.description}"),
        console=console,
    ) as progress:
        # Load model
        task = progress.add_task("Loading model...", total=None)
        analyzer = Sharingan(model)
        analyzer.load()
        progress.update(task, description="Model loaded")

        # Analyze
        progress.update(task, description="Analyzing attention...")
        result = analyzer.analyze(
            prompt,
            generate=generate,
            max_new_tokens=max_tokens,
        )
        progress.update(task, description="Analysis complete")

    # Display summary
    summary = result.summary()
    table = Table(title="Attention Analysis", border_style="red")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="red")

    table.add_row("Model", summary["model"])
    table.add_row("Layers", str(summary["num_layers"]))
    table.add_row("Heads", str(summary["num_heads"]))
    table.add_row("Sequence Length", str(summary["seq_len"]))
    table.add_row("Mean Entropy", f"{summary['mean_entropy']:.4f}")
    table.add_row("Attention Sinks", str(summary["num_sinks"]))

    console.print(table)

    # Show tokens
    tokens_str = " ".join(result.tokens[:50])
    if len(result.tokens) > 50:
        tokens_str += f" ... (+{len(result.tokens) - 50} more)"
    console.print(Panel(tokens_str, title="Tokens", border_style="red"))

    # Output
    if output:
        output = Path(output)
        if output.suffix == ".html":
            result.to_html(str(output))
            console.print(f"[green]✓[/green] Exported to {output}")
        else:
            fig = result.plot(layer=layer, head=head, interactive=interactive)
            if hasattr(fig, "savefig"):
                fig.savefig(str(output), dpi=150, bbox_inches="tight")
            else:
                fig.write_image(str(output))
            console.print(f"[green]✓[/green] Saved to {output}")

    if show and not output:
        if interactive:
            fig = result.plot(layer=layer, head=head, interactive=True)
            fig.show()
        else:
            import matplotlib.pyplot as plt

            result.plot(layer=layer, head=head)
            plt.show()


@app.command()
def dashboard(
    port: int = typer.Option(
        7860,
        "--port",
        "-p",
        help="Port to run dashboard on",
    ),
    share: bool = typer.Option(
        False,
        "--share",
        "-s",
        help="Create public link",
    ),
):
    """Launch interactive Gradio dashboard."""
    from sharingan.ui.dashboard import launch_dashboard

    console.print(
        Panel(
            f"[bold red]Sharingan Dashboard[/bold red]\n"
            f"Running on http://localhost:{port}",
            border_style="red",
        )
    )

    launch_dashboard(port=port, share=share)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model name to get info about"),
):
    """Get information about a model's architecture."""
    from sharingan import Sharingan

    with Progress(
        SpinnerColumn(style="red"),
        TextColumn("[bold red]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        analyzer = Sharingan(model)
        analyzer.load()

    adapter = analyzer.adapter
    config = analyzer.model.config

    table = Table(title=f"Model: {model}", border_style="red")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="red")

    table.add_row("Architecture", getattr(config, "architectures", ["Unknown"])[0])
    table.add_row("Layers", str(adapter.num_layers))
    table.add_row("Query Heads", str(adapter.num_heads))
    table.add_row("KV Heads", str(adapter.num_kv_heads))
    table.add_row("Uses GQA", "Yes" if adapter.uses_gqa else "No")
    if adapter.uses_gqa:
        table.add_row("GQA Ratio", f"{adapter.gqa_ratio}:1")
    table.add_row("Hidden Size", str(getattr(config, "hidden_size", "N/A")))
    table.add_row("Vocab Size", str(getattr(config, "vocab_size", "N/A")))

    console.print(table)


@app.command()
def version():
    """Show version information."""
    from sharingan import __version__

    console.print(f"[bold red]Sharingan[/bold red] v{__version__}")


if __name__ == "__main__":
    app()
