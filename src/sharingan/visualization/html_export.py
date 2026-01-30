"""Standalone HTML export for attention visualizations."""

from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sharingan.core.result import AttentionResult


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sharingan - Attention Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary: #B91C1C;
            --secondary: #1F2937;
            --accent: #EF4444;
            --background: #111827;
            --surface: #1F2937;
            --text: #F9FAFB;
            --text-muted: #9CA3AF;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--background);
            color: var(--text);
            min-height: 100vh;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, #7F1D1D 100%);
            padding: 1.5rem 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .header .subtitle {{
            color: rgba(255,255,255,0.8);
            font-size: 0.875rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--secondary);
            padding-bottom: 0.5rem;
        }}

        .tab {{
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            border-radius: 0.5rem 0.5rem 0 0;
            transition: all 0.2s;
        }}

        .tab:hover {{
            color: var(--text);
            background: var(--surface);
        }}

        .tab.active {{
            color: var(--text);
            background: var(--primary);
        }}

        .panel {{
            display: none;
            background: var(--surface);
            border-radius: 0.5rem;
            padding: 1.5rem;
        }}

        .panel.active {{
            display: block;
        }}

        .controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .control-group label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        .control-group select,
        .control-group input {{
            padding: 0.5rem;
            background: var(--background);
            border: 1px solid var(--secondary);
            color: var(--text);
            border-radius: 0.25rem;
            min-width: 120px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .metric-card {{
            background: var(--background);
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 3px solid var(--primary);
        }}

        .metric-card .value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
        }}

        .metric-card .label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        .plot-container {{
            background: var(--background);
            border-radius: 0.5rem;
            overflow: hidden;
        }}

        .tokens-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
            margin-top: 1rem;
            max-height: 150px;
            overflow-y: auto;
        }}

        .token {{
            padding: 0.25rem 0.5rem;
            background: var(--background);
            border-radius: 0.25rem;
            font-family: monospace;
            font-size: 0.75rem;
        }}

        .token.highlight {{
            background: var(--primary);
        }}

        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.75rem;
        }}

        .footer a {{
            color: var(--accent);
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Sharingan</h1>
            <div class="subtitle">{model_name}</div>
        </div>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">Overview</button>
            <button class="tab" onclick="showTab('layers')">By Layer</button>
            <button class="tab" onclick="showTab('metrics')">Metrics</button>
            <button class="tab" onclick="showTab('tokens')">Tokens</button>
        </div>

        <div id="overview" class="panel active">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="value">{num_layers}</div>
                    <div class="label">Layers</div>
                </div>
                <div class="metric-card">
                    <div class="value">{num_heads}</div>
                    <div class="label">Heads</div>
                </div>
                <div class="metric-card">
                    <div class="value">{seq_len}</div>
                    <div class="label">Sequence Length</div>
                </div>
                <div class="metric-card">
                    <div class="value">{mean_entropy:.3f}</div>
                    <div class="label">Mean Entropy</div>
                </div>
            </div>
            <div class="plot-container" id="overview-plot"></div>
        </div>

        <div id="layers" class="panel">
            <div class="controls">
                <div class="control-group">
                    <label>Layer</label>
                    <select id="layer-select" onchange="updateLayerPlot()">
                        {layer_options}
                    </select>
                </div>
                <div class="control-group">
                    <label>Head</label>
                    <select id="head-select" onchange="updateLayerPlot()">
                        <option value="all">All (mean)</option>
                        {head_options}
                    </select>
                </div>
            </div>
            <div class="plot-container" id="layer-plot"></div>
        </div>

        <div id="metrics" class="panel">
            <div class="plot-container" id="entropy-plot"></div>
            <div class="plot-container" id="importance-plot" style="margin-top: 1rem;"></div>
        </div>

        <div id="tokens" class="panel">
            <h3 style="margin-bottom: 1rem;">Input Tokens ({num_tokens})</h3>
            <div class="tokens-list">
                {tokens_html}
            </div>
            {sinks_html}
        </div>
    </div>

    <div class="footer">
        Generated by <a href="https://github.com/sharingan-viz/sharingan">Sharingan</a>
    </div>

    <script>
        const data = {data_json};

        const colorscale = [
            [0.0, '#111827'],
            [0.2, '#1F2937'],
            [0.4, '#7F1D1D'],
            [0.6, '#B91C1C'],
            [0.8, '#EF4444'],
            [1.0, '#FCA5A5']
        ];

        const layout = {{
            paper_bgcolor: '#111827',
            plot_bgcolor: '#111827',
            font: {{ color: '#F9FAFB' }},
            margin: {{ t: 40, r: 20, b: 40, l: 40 }}
        }};

        function showTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${{tabId}}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}

        function plotOverview() {{
            const trace = {{
                z: data.global_attention,
                type: 'heatmap',
                colorscale: colorscale,
                hovertemplate: 'Query: %{{y}}<br>Key: %{{x}}<br>Attention: %{{z:.4f}}<extra></extra>'
            }};
            Plotly.newPlot('overview-plot', [trace], {{
                ...layout,
                title: 'Global Attention View',
                xaxis: {{ title: 'Key Position' }},
                yaxis: {{ title: 'Query Position', autorange: 'reversed' }},
                height: 600
            }});
        }}

        function updateLayerPlot() {{
            const layer = document.getElementById('layer-select').value;
            const head = document.getElementById('head-select').value;

            let attn;
            if (head === 'all') {{
                // Mean across heads
                attn = data.attention[layer].reduce((acc, h) => {{
                    return acc.map((row, i) => row.map((v, j) => v + h[i][j]));
                }}, data.attention[layer][0].map(row => row.map(() => 0)));
                attn = attn.map(row => row.map(v => v / data.attention[layer].length));
            }} else {{
                attn = data.attention[layer][head];
            }}

            const trace = {{
                z: attn,
                type: 'heatmap',
                colorscale: colorscale,
                hovertemplate: 'Query: %{{y}}<br>Key: %{{x}}<br>Attention: %{{z:.4f}}<extra></extra>'
            }};

            Plotly.newPlot('layer-plot', [trace], {{
                ...layout,
                title: `Layer ${{layer}}${{head !== 'all' ? ', Head ' + head : ' (all heads)'}}`,
                xaxis: {{ title: 'Key Position' }},
                yaxis: {{ title: 'Query Position', autorange: 'reversed' }},
                height: 600
            }});
        }}

        function plotMetrics() {{
            // Entropy plot
            const entropyTrace = {{
                y: data.entropy,
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                line: {{ color: '#EF4444' }},
                fillcolor: 'rgba(239, 68, 68, 0.3)'
            }};
            Plotly.newPlot('entropy-plot', [entropyTrace], {{
                ...layout,
                title: 'Attention Entropy by Position',
                xaxis: {{ title: 'Position' }},
                yaxis: {{ title: 'Entropy' }},
                height: 300
            }});

            // Importance plot
            const importanceTrace = {{
                y: data.importance.slice(0, 50),
                x: data.tokens.slice(0, 50),
                type: 'bar',
                marker: {{ color: '#B91C1C' }}
            }};
            Plotly.newPlot('importance-plot', [importanceTrace], {{
                ...layout,
                title: 'Token Importance (first 50)',
                xaxis: {{ title: 'Token' }},
                yaxis: {{ title: 'Importance' }},
                height: 300
            }});
        }}

        // Initialize
        plotOverview();
        updateLayerPlot();
        plotMetrics();
    </script>
</body>
</html>
"""


def export_html(
    result: "AttentionResult",
    path: str,
    include_metrics: bool = True,
    max_attention_size: int = 128,
) -> None:
    """Export visualization to standalone HTML file.

    Args:
        result: AttentionResult object
        path: Output file path
        include_metrics: Whether to include metrics
        max_attention_size: Max size for attention data in HTML
    """
    from sharingan.attention.downsampler import downsample_attention

    # Prepare data for JSON embedding
    # Downsample attention for reasonable file size
    attention_data = {}
    for layer in range(result.num_layers):
        attention_data[layer] = {}
        for head in range(result.num_heads):
            attn = result.attention[layer, head]
            if attn.shape[0] > max_attention_size:
                attn = downsample_attention(attn, target_size=max_attention_size)
            attention_data[layer][head] = attn.tolist()

    # Global view
    global_attn = result.get_attention(aggregate="mean")
    if global_attn.shape[0] > 256:
        global_attn = downsample_attention(global_attn, target_size=256)

    # Metrics
    entropy = result.attention_entropy().tolist()
    importance = result.token_importance().tolist()
    summary = result.summary()

    # Prepare data JSON
    data = {
        "attention": attention_data,
        "global_attention": global_attn.tolist(),
        "tokens": result.tokens,
        "entropy": entropy,
        "importance": importance,
    }

    # Generate HTML elements
    layer_options = "\n".join(
        f'<option value="{i}">Layer {i}</option>' for i in range(result.num_layers)
    )
    head_options = "\n".join(
        f'<option value="{i}">Head {i}</option>' for i in range(result.num_heads)
    )

    tokens_html = "".join(
        f'<span class="token">{_escape_html(t)}</span>' for t in result.tokens[:200]
    )
    if len(result.tokens) > 200:
        tokens_html += f'<span class="token">... +{len(result.tokens) - 200} more</span>'

    # Sinks info
    sinks = result.attention_sinks()
    if sinks:
        sinks_html = '<h3 style="margin-top: 1.5rem; margin-bottom: 0.5rem;">Attention Sinks</h3><ul>'
        for sink in sinks[:5]:
            sinks_html += (
                f'<li style="color: var(--text-muted); margin: 0.25rem 0;">'
                f'Position {sink["position"]}: "{_escape_html(sink["token"])}" '
                f'(attention: {sink["attention_received"]:.3f})</li>'
            )
        sinks_html += "</ul>"
    else:
        sinks_html = ""

    # Fill template
    html = HTML_TEMPLATE.format(
        model_name=result.model_name,
        num_layers=result.num_layers,
        num_heads=result.num_heads,
        seq_len=result.seq_len,
        mean_entropy=summary["mean_entropy"],
        num_tokens=len(result.tokens),
        layer_options=layer_options,
        head_options=head_options,
        tokens_html=tokens_html,
        sinks_html=sinks_html,
        data_json=json.dumps(data),
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
