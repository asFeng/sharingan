"""Basic usage examples for Sharingan."""

from sharingan import Sharingan, visualize


def simple_visualization():
    """One-liner visualization."""
    # The simplest way to visualize attention
    result = visualize(
        "Qwen/Qwen3-0.6B",
        "The capital of France is",
        show=True,
    )
    print(f"Analyzed {result.seq_len} tokens across {result.num_layers} layers")


def advanced_usage():
    """More detailed analysis."""
    # Create analyzer
    analyzer = Sharingan("Qwen/Qwen3-0.6B")

    # Analyze prompt
    result = analyzer.analyze(
        "The quick brown fox jumps over the lazy dog.",
        generate=False,
    )

    # Explore results
    print(f"Model: {result.model_name}")
    print(f"Layers: {result.num_layers}")
    print(f"Heads: {result.num_heads}")
    print(f"Sequence length: {result.seq_len}")
    print()

    # Entropy analysis
    entropy = result.attention_entropy()
    print(f"Mean entropy: {entropy.mean():.4f}")
    print(f"Min entropy: {entropy.min():.4f} at position {entropy.argmin()}")
    print(f"Max entropy: {entropy.max():.4f} at position {entropy.argmax()}")
    print()

    # Attention sinks
    sinks = result.attention_sinks()
    print(f"Found {len(sinks)} attention sinks:")
    for sink in sinks[:3]:
        print(f"  Position {sink['position']}: '{sink['token']}' (attn: {sink['attention_received']:.4f})")
    print()

    # Token importance
    importance = result.token_importance()
    top_indices = importance.argsort()[-5:][::-1]
    print("Most important tokens:")
    for idx in top_indices:
        print(f"  {idx}: '{result.tokens[idx]}' (importance: {importance[idx]:.4f})")
    print()

    # Hotspots
    hotspots = result.hotspots(top_k=5)
    print("Top attention hotspots:")
    for hs in hotspots:
        print(f"  '{hs['from_token']}' -> '{hs['to_token']}' (attn: {hs['attention']:.4f})")


def plot_examples():
    """Visualization examples."""
    analyzer = Sharingan("Qwen/Qwen3-0.6B")
    result = analyzer.analyze("Hello, how are you today?")

    # Basic heatmap
    fig = result.plot()
    fig.savefig("attention_basic.png", dpi=150)

    # Specific layer/head
    fig = result.plot(layer=5, head=3)
    fig.savefig("attention_layer5_head3.png", dpi=150)

    # Interactive plot
    fig = result.plot(interactive=True)
    fig.write_html("attention_interactive.html")

    # Export to standalone HTML
    result.to_html("attention_full.html")


def generation_example():
    """Analyze attention during generation."""
    analyzer = Sharingan("Qwen/Qwen3-0.6B")

    result = analyzer.analyze(
        "Once upon a time",
        generate=True,
        max_new_tokens=20,
    )

    print(f"Prompt tokens: {result.prompt_length}")
    print(f"Generated tokens: {result.generated_length}")
    print(f"Total sequence: {result.seq_len}")
    print()
    print("Generated text:")
    print("".join(result.tokens))


if __name__ == "__main__":
    print("=" * 50)
    print("Advanced Usage Example")
    print("=" * 50)
    advanced_usage()
