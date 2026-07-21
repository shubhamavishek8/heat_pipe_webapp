"""Page 9 - Authors: attribution for the study behind this application."""
import streamlit as st

from app_utils import (inject_css, page_header, C_PRIMARY, TEXT, MUTED,
                       CARD_BG, BORDER, FONT_TNR)

st.set_page_config(page_title="Authors", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()

page_header("Authors",
            "The people behind this study and application.")


def author_card(initials, name, role, affiliation):
    role_html = (f"<div style='color:{MUTED};font-size:0.95rem;margin-top:2px'>{role}</div>"
                 if role else "")
    st.markdown(
        f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;"
        f"padding:22px 24px;margin:8px 0;display:flex;align-items:center;gap:20px'>"
        f"<div style='min-width:64px;height:64px;border-radius:50%;background:{C_PRIMARY};"
        f"color:#ffffff;display:flex;align-items:center;justify-content:center;"
        f"font-size:1.5rem;font-weight:700;font-family:{FONT_TNR}'>{initials}</div>"
        f"<div>"
        f"<div style='font-size:1.25rem;font-weight:700;color:{TEXT}'>{name}</div>"
        f"{role_html}"
        f"<div style='color:{MUTED};font-size:0.95rem;margin-top:4px'>{affiliation}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


affil = ("Department of Mechanical Engineering, National Institute of Technology "
         "Meghalaya, Sohra, Meghalaya, India - 793108")

author_card("SA", "Shubhamshree Avishek", "Research Scholar", affil)
author_card("KD", "Dr. Koushik Das", "Associate Professor", affil)

st.markdown("---")
st.markdown(
    "<div style='color:" + TEXT + "'>"
    "<b>About this work.</b> This application accompanies a study on surrogate-based "
    "design of heat pipes: a finite-element thermal-hydraulic model is emulated by "
    "machine-learning surrogates (selected by leave-one-out cross-validation), enabling "
    "instant prediction, constrained design optimisation, manufacturing-tolerance "
    "propagation, uncertainty mapping and active-learning guidance without re-running "
    "the solver.</div>",
    unsafe_allow_html=True,
)
st.caption("Built with Streamlit. The surrogate, scalers and dataset are produced by the "
           "authors' offline training pipeline and loaded here without retraining.")
