"""
╔══════════════════════════════════════════════════════════════╗
║  STREAMLIT DESIGN SYSTEM — Dark & Sleek Analytics Dashboard  ║
║  Based on UI UX Pro Max design intelligence principles       ║
║                                                              ║
║  HOW TO USE:                                                 ║
║  1. Drop this file in your project folder                    ║
║  2. In your main app.py, add at the very top:                ║
║     from style_inject import inject_styles                   ║
║     inject_styles()                                          ║
║  3. Use the helper functions for styled components           ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st


# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
TOKENS = {
    # Background layers (darkest → lightest)
    "bg_base": "#0B0F19",        # Page background
    "bg_surface": "#111827",     # Cards, containers
    "bg_elevated": "#1A2236",    # Elevated panels, modals
    "bg_hover": "#1E293B",       # Hover states

    # Accent colors
    "accent_primary": "#6366F1",    # Indigo: primary actions
    "accent_secondary": "#8B5CF6",  # Purple: secondary elements
    "accent_success": "#10B981",    # Emerald: positive states
    "accent_warning": "#F59E0B",    # Amber: warnings
    "accent_danger": "#EF4444",     # Red: errors/alerts
    "accent_info": "#3B82F6",       # Blue: informational

    # Text hierarchy
    "text_primary": "#F1F5F9",      # Main text
    "text_secondary": "#94A3B8",    # Subdued text
    "text_muted": "#475569",        # Hints, placeholders
    "text_on_accent": "#FFFFFF",    # Text on accent backgrounds

    # Borders
    "border_subtle": "rgba(148, 163, 184, 0.08)",
    "border_default": "rgba(148, 163, 184, 0.12)",
    "border_focus": "rgba(99, 102, 241, 0.5)",

    # Spacing scale (rem)
    "space_xs": "0.25rem",
    "space_sm": "0.5rem",
    "space_md": "1rem",
    "space_lg": "1.5rem",
    "space_xl": "2rem",
    "space_2xl": "3rem",

    # Typography
    "font_display": "'DM Sans', sans-serif",
    "font_body": "'DM Sans', sans-serif",
    "font_mono": "'JetBrains Mono', monospace",
    "text_xs": "0.75rem",
    "text_sm": "0.875rem",
    "text_base": "1rem",
    "text_lg": "1.125rem",
    "text_xl": "1.25rem",
    "text_2xl": "1.5rem",
    "text_3xl": "2rem",

    # Radius
    "radius_sm": "6px",
    "radius_md": "10px",
    "radius_lg": "14px",
    "radius_xl": "20px",

    # Shadows
    "shadow_sm": "0 1px 3px rgba(0,0,0,0.4)",
    "shadow_md": "0 4px 12px rgba(0,0,0,0.5)",
    "shadow_lg": "0 12px 40px rgba(0,0,0,0.6)",
    "shadow_glow": "0 0 20px rgba(99,102,241,0.15)",
}


def inject_styles():
    """
    Call this at the very top of your app (after st.set_page_config).
    Injects the full design system CSS into your Streamlit app.
    """

    # ── Page config suggestion ──
    # Make sure you have this BEFORE calling inject_styles():
    # st.set_page_config(
    #     page_title="Your Dashboard",
    #     page_icon="📊",
    #     layout="wide",
    #     initial_sidebar_state="expanded"
    # )

    css = f"""
    <style>
    /* ═══════════════════════════════════════════
       GOOGLE FONTS
       ═══════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ═══════════════════════════════════════════
       BASE / RESET
       ═══════════════════════════════════════════ */
    .stApp {{
        background-color: {TOKENS["bg_base"]};
        font-family: {TOKENS["font_body"]};
        color: {TOKENS["text_primary"]};
    }}

    /* Remove default top padding */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }}

    /* ═══════════════════════════════════════════
       TYPOGRAPHY
       ═══════════════════════════════════════════ */
    h1 {{
        font-family: {TOKENS["font_display"]} !important;
        font-weight: 700 !important;
        color: {TOKENS["text_primary"]} !important;
        font-size: {TOKENS["text_3xl"]} !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.25rem !important;
    }}

    h2 {{
        font-family: {TOKENS["font_display"]} !important;
        font-weight: 600 !important;
        color: {TOKENS["text_primary"]} !important;
        font-size: {TOKENS["text_2xl"]} !important;
        letter-spacing: -0.01em !important;
    }}

    h3 {{
        font-family: {TOKENS["font_display"]} !important;
        font-weight: 600 !important;
        color: {TOKENS["text_secondary"]} !important;
        font-size: {TOKENS["text_lg"]} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em !important;
    }}

    p, span, label, .stMarkdown {{
        font-family: {TOKENS["font_body"]} !important;
        color: {TOKENS["text_secondary"]};
    }}

    /* ═══════════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════════ */
    section[data-testid="stSidebar"] {{
        background-color: {TOKENS["bg_surface"]} !important;
        border-right: 1px solid {TOKENS["border_subtle"]} !important;
    }}

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {TOKENS["text_primary"]} !important;
    }}

    /* ═══════════════════════════════════════════
       METRIC CARDS (st.metric)
       ═══════════════════════════════════════════ */
    div[data-testid="stMetric"] {{
        background: {TOKENS["bg_surface"]};
        border: 1px solid {TOKENS["border_default"]};
        border-radius: {TOKENS["radius_lg"]};
        padding: 1.25rem 1.5rem;
        box-shadow: {TOKENS["shadow_sm"]};
        transition: all 0.2s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: {TOKENS["border_focus"]};
        box-shadow: {TOKENS["shadow_glow"]};
        transform: translateY(-1px);
    }}

    div[data-testid="stMetric"] label {{
        color: {TOKENS["text_secondary"]} !important;
        font-size: {TOKENS["text_sm"]} !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-family: {TOKENS["font_display"]} !important;
        font-weight: 700 !important;
        color: {TOKENS["text_primary"]} !important;
        font-size: {TOKENS["text_2xl"]} !important;
    }}

    div[data-testid="stMetricDelta"] {{
        font-family: {TOKENS["font_mono"]} !important;
        font-weight: 500 !important;
    }}

    /* ═══════════════════════════════════════════
       FORMS & INPUTS
       ═══════════════════════════════════════════ */
    /* Text inputs, number inputs, text areas */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stTextArea > div > div {{
        background-color: {TOKENS["bg_elevated"]} !important;
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_md"]} !important;
        color: {TOKENS["text_primary"]} !important;
        font-family: {TOKENS["font_body"]} !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within,
    .stTextArea > div > div:focus-within {{
        border-color: {TOKENS["accent_primary"]} !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }}

    input, textarea {{
        color: {TOKENS["text_primary"]} !important;
        font-family: {TOKENS["font_body"]} !important;
        caret-color: {TOKENS["accent_primary"]} !important;
    }}

    input::placeholder, textarea::placeholder {{
        color: {TOKENS["text_muted"]} !important;
        font-style: italic;
    }}

    /* Select boxes / dropdowns */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: {TOKENS["bg_elevated"]} !important;
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_md"]} !important;
        color: {TOKENS["text_primary"]} !important;
    }}

    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {{
        border-color: {TOKENS["accent_primary"]} !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }}

    /* Slider */
    .stSlider > div > div > div {{
        background-color: {TOKENS["bg_hover"]} !important;
    }}

    .stSlider [data-testid="stThumbValue"] {{
        color: {TOKENS["text_primary"]} !important;
        font-family: {TOKENS["font_mono"]} !important;
    }}

    /* Date/time inputs */
    .stDateInput > div > div,
    .stTimeInput > div > div {{
        background-color: {TOKENS["bg_elevated"]} !important;
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_md"]} !important;
    }}

    /* Checkbox & radio */
    .stCheckbox label, .stRadio label {{
        color: {TOKENS["text_secondary"]} !important;
        font-family: {TOKENS["font_body"]} !important;
    }}

    /* Form labels */
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stDateInput label,
    .stTimeInput label,
    .stRadio label p,
    .stCheckbox label span {{
        color: {TOKENS["text_secondary"]} !important;
        font-weight: 500 !important;
        font-size: {TOKENS["text_sm"]} !important;
        margin-bottom: 0.25rem !important;
    }}

    /* ═══════════════════════════════════════════
       BUTTONS
       ═══════════════════════════════════════════ */
    /* Primary button */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {{
        background: linear-gradient(135deg, {TOKENS["accent_primary"]}, {TOKENS["accent_secondary"]}) !important;
        color: {TOKENS["text_on_accent"]} !important;
        border: none !important;
        border-radius: {TOKENS["radius_md"]} !important;
        font-family: {TOKENS["font_body"]} !important;
        font-weight: 600 !important;
        font-size: {TOKENS["text_sm"]} !important;
        padding: 0.6rem 1.5rem !important;
        letter-spacing: 0.02em;
        transition: all 0.2s ease !important;
        box-shadow: {TOKENS["shadow_sm"]} !important;
    }}

    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {{
        box-shadow: 0 0 24px rgba(99, 102, 241, 0.35) !important;
        transform: translateY(-1px);
    }}

    /* Secondary button */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind]) {{
        background-color: transparent !important;
        color: {TOKENS["text_secondary"]} !important;
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_md"]} !important;
        font-family: {TOKENS["font_body"]} !important;
        font-weight: 500 !important;
        font-size: {TOKENS["text_sm"]} !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind]):hover {{
        background-color: {TOKENS["bg_hover"]} !important;
        border-color: {TOKENS["text_muted"]} !important;
        color: {TOKENS["text_primary"]} !important;
    }}

    /* ═══════════════════════════════════════════
       TABLES & DATAFRAMES
       ═══════════════════════════════════════════ */
    .stDataFrame {{
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_lg"]} !important;
        overflow: hidden;
    }}

    .stDataFrame [data-testid="glideDataEditor"] {{
        border-radius: {TOKENS["radius_lg"]} !important;
    }}

    /* ═══════════════════════════════════════════
       CHARTS (Plotly / Altair / Matplotlib)
       ═══════════════════════════════════════════ */
    .stPlotlyChart, .stVegaLiteChart {{
        background: {TOKENS["bg_surface"]} !important;
        border: 1px solid {TOKENS["border_subtle"]} !important;
        border-radius: {TOKENS["radius_lg"]} !important;
        padding: 0.75rem !important;
        box-shadow: {TOKENS["shadow_sm"]};
    }}

    /* ═══════════════════════════════════════════
       TABS
       ═══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem;
        background: {TOKENS["bg_surface"]};
        border-radius: {TOKENS["radius_md"]};
        padding: 4px;
        border: 1px solid {TOKENS["border_subtle"]};
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: {TOKENS["radius_sm"]} !important;
        color: {TOKENS["text_muted"]} !important;
        font-family: {TOKENS["font_body"]} !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {TOKENS["accent_primary"]} !important;
        color: {TOKENS["text_on_accent"]} !important;
    }}

    /* ═══════════════════════════════════════════
       EXPANDERS
       ═══════════════════════════════════════════ */
    .streamlit-expanderHeader {{
        background-color: {TOKENS["bg_surface"]} !important;
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_md"]} !important;
        color: {TOKENS["text_primary"]} !important;
        font-weight: 500 !important;
    }}

    /* ═══════════════════════════════════════════
       ALERTS / CALLOUTS
       ═══════════════════════════════════════════ */
    .stAlert {{
        border-radius: {TOKENS["radius_md"]} !important;
        font-family: {TOKENS["font_body"]} !important;
    }}

    /* ═══════════════════════════════════════════
       DIVIDERS
       ═══════════════════════════════════════════ */
    hr {{
        border-color: {TOKENS["border_subtle"]} !important;
        margin: {TOKENS["space_lg"]} 0 !important;
    }}

    /* ═══════════════════════════════════════════
       SCROLLBAR (Webkit)
       ═══════════════════════════════════════════ */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {TOKENS["bg_base"]};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {TOKENS["text_muted"]};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {TOKENS["text_secondary"]};
    }}

    /* ═══════════════════════════════════════════
       FORM CONTAINER (st.form)
       ═══════════════════════════════════════════ */
    [data-testid="stForm"] {{
        background: {TOKENS["bg_surface"]} !important;
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_lg"]} !important;
        padding: 1.5rem !important;
        box-shadow: {TOKENS["shadow_sm"]};
    }}

    /* ═══════════════════════════════════════════
       TOOLTIPS & HELP TEXT
       ═══════════════════════════════════════════ */
    .stTooltipIcon {{
        color: {TOKENS["text_muted"]} !important;
    }}

    /* ═══════════════════════════════════════════
       TOAST / NOTIFICATIONS
       ═══════════════════════════════════════════ */
    .stToast {{
        background: {TOKENS["bg_elevated"]} !important;
        border: 1px solid {TOKENS["border_default"]} !important;
        border-radius: {TOKENS["radius_md"]} !important;
    }}

    /* ═══════════════════════════════════════════
       ANIMATIONS
       ═══════════════════════════════════════════ */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    .stMetric, [data-testid="stForm"], .stPlotlyChart {{
        animation: fadeIn 0.4s ease-out;
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────────

def styled_header(title: str, subtitle: str = ""):
    """
    Renders a styled page header with optional subtitle.

    Usage:
        styled_header("Revenue Dashboard", "Real-time analytics overview")
    """
    html = f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="
            font-family: {TOKENS["font_display"]};
            font-weight: 700;
            font-size: 2rem;
            color: {TOKENS["text_primary"]};
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
        ">{title}</h1>
        {"<p style='font-size: 1rem; color: " + TOKENS["text_secondary"] + "; margin-top: 0.35rem; font-family: " + TOKENS["font_body"] + ";'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_card(content: str, accent_color: str = None):
    """
    Renders content inside a styled card with optional left accent border.

    Usage:
        styled_card("This is important info", accent_color="#6366F1")
    """
    accent = f"border-left: 3px solid {accent_color};" if accent_color else ""
    html = f"""
    <div style="
        background: {TOKENS["bg_surface"]};
        border: 1px solid {TOKENS["border_default"]};
        border-radius: {TOKENS["radius_lg"]};
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: {TOKENS["shadow_sm"]};
        {accent}
    ">
        <span style="color: {TOKENS["text_secondary"]}; font-family: {TOKENS["font_body"]}; font-size: 0.95rem; line-height: 1.6;">
            {content}
        </span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_kpi(label: str, value: str, delta: str = "", delta_color: str = ""):
    """
    Renders a KPI card with label, big value, and optional delta.

    Usage:
        styled_kpi("Total Revenue", "$1.2M", delta="+12.5%", delta_color="#10B981")
    """
    delta_html = ""
    if delta:
        dc = delta_color or TOKENS["accent_success"]
        delta_html = f'<span style="font-family: {TOKENS["font_mono"]}; font-size: 0.8rem; color: {dc}; font-weight: 500;">{delta}</span>'

    html = f"""
    <div style="
        background: {TOKENS["bg_surface"]};
        border: 1px solid {TOKENS["border_default"]};
        border-radius: {TOKENS["radius_lg"]};
        padding: 1.25rem 1.5rem;
        box-shadow: {TOKENS["shadow_sm"]};
        transition: all 0.2s ease;
    ">
        <div style="
            font-size: 0.75rem;
            color: {TOKENS["text_muted"]};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            font-family: {TOKENS["font_body"]};
            margin-bottom: 0.5rem;
        ">{label}</div>
        <div style="display: flex; align-items: baseline; gap: 0.75rem;">
            <span style="
                font-family: {TOKENS["font_display"]};
                font-size: 1.75rem;
                font-weight: 700;
                color: {TOKENS["text_primary"]};
                letter-spacing: -0.02em;
            ">{value}</span>
            {delta_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_divider():
    """Renders a subtle horizontal divider."""
    st.markdown(
        f'<hr style="border: none; border-top: 1px solid {TOKENS["border_subtle"]}; margin: 1.5rem 0;">',
        unsafe_allow_html=True
    )


def styled_section_label(text: str):
    """
    Renders a small uppercase section label.

    Usage:
        styled_section_label("Filter Options")
    """
    html = f"""
    <div style="
        font-size: 0.7rem;
        color: {TOKENS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        font-family: {TOKENS["font_body"]};
        margin-bottom: 0.75rem;
        margin-top: 1rem;
    ">{text}</div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY THEME (use with your charts)
# ─────────────────────────────────────────────

PLOTLY_THEME = dict(
    template="plotly_dark",
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color=TOKENS["text_secondary"],
            size=13,
        ),
        title=dict(
            font=dict(
                family="DM Sans, sans-serif",
                size=16,
                color=TOKENS["text_primary"],
            ),
            x=0,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.06)",
            linecolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.06)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.06)",
            linecolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.06)",
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=[
            TOKENS["accent_primary"],   # Indigo
            TOKENS["accent_secondary"], # Purple
            TOKENS["accent_info"],      # Blue
            TOKENS["accent_success"],   # Emerald
            TOKENS["accent_warning"],   # Amber
            TOKENS["accent_danger"],    # Red
            "#EC4899",                  # Pink
            "#14B8A6",                  # Teal
        ],
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TOKENS["text_secondary"], size=12),
        ),
    ),
)

# Convenience: apply to a Plotly figure
def apply_plotly_theme(fig):
    """
    Apply the dark sleek theme to any Plotly figure.

    Preserves any existing title text. The theme only sets font/position
    defaults: it never overwrites the title string.

    Usage:
        import plotly.express as px
        fig = px.line(df, x="date", y="revenue")
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    """
    # Extract theme layout without the title key (to avoid overwriting title text)
    theme = {k: v for k, v in PLOTLY_THEME["layout"].items() if k != "title"}
    fig.update_layout(**theme)

    # Apply title font styling while preserving existing title text
    existing_title = fig.layout.title
    if existing_title and existing_title.text:
        fig.update_layout(title=dict(
            text=existing_title.text,
            font=PLOTLY_THEME["layout"]["title"]["font"],
            x=PLOTLY_THEME["layout"]["title"]["x"],
            xanchor=PLOTLY_THEME["layout"]["title"]["xanchor"],
        ))
    else:
        # No title set — ensure we don't show "undefined"
        fig.update_layout(title=dict(text=""))

    return fig


# ─────────────────────────────────────────────
# STREAMLIT THEME CONFIG (for .streamlit/config.toml)
# ─────────────────────────────────────────────
TOML_THEME = """
# ───────────────────────────────────────────
# .streamlit/config.toml
# Copy this file to your project's .streamlit/ folder
# ───────────────────────────────────────────

[theme]
base = "dark"
primaryColor = "#6366F1"
backgroundColor = "#0B0F19"
secondaryBackgroundColor = "#111827"
textColor = "#F1F5F9"
font = "sans serif"
"""
