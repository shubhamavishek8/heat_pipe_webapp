"""Page 08 - Authors: attribution and an overview of the study behind this application."""
import streamlit as st

import core
from app_utils import (get_assets, inject_css, page_header, C_PRIMARY, TEXT, MUTED,
                       CARD_BG, BORDER, FONT_TNR, SYM)

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

# ----------------------------------------------------------------------------- #
# About this work - figures are read from the loaded artifacts so the text stays
# correct if the surrogate or dataset is replaced.
# ----------------------------------------------------------------------------- #
A = get_assets()
lo_vp, hi_vp = core.BOUNDS["vp_vs"]
lo_po, hi_po = core.BOUNDS["po"]
_m = A.manifest
_r2 = _m.get("loocv_overall_r2")
_r2_txt = f" (leave-one-out R\u00b2 = {_r2:.3f})" if _r2 is not None else ""

st.subheader("About this work")

st.markdown(
    f"""
**Motivation.** Flat heat pipes transport heat by evaporation and capillary-driven
condensate return, so their performance is governed by a competition rather than a
single effect: a wick that returns liquid readily tends to raise the thermal
resistance, while a wick tuned for low resistance tends to throttle the flow and
raise the pressure drop the capillary structure must overcome. Resolving that
trade-off with high-fidelity simulation alone is expensive, because each candidate
design requires a fresh coupled thermal-hydraulic solution. This work addresses the
cost directly: a validated finite-element (FE) model supplies a designed set of
simulations, machine-learning surrogates learn the input-output mapping from those
simulations, and the surrogates then stand in for the solver during design search,
sensitivity study and robustness assessment.
""", unsafe_allow_html=True)

st.markdown(
    f"""
**Physical problem and design space.** The study concerns a hybrid wick architecture
in which a primary and a secondary wick act together, and it characterises a design by
two dimensionless parameters: the primary-to-secondary wick volume ratio
{SYM['vp_vs']}, which sets how the available wick volume is distributed between the two
structures, and the wick porosity {SYM['po']}, which governs permeability and effective
conductivity. Two responses summarise performance: the effective thermal resistance
{SYM['r_th']} (K/W), to be minimised, and the total pressure drop {SYM['p_tot']} (Pa),
which is constrained because the capillary structure can only sustain a finite driving
head. The FE campaign spans
{SYM['vp_vs']} \u2208 [{lo_vp:.3f}, {hi_vp:.3f}] and
{SYM['po']} \u2208 [{lo_po:.3f}, {hi_po:.3f}], and comprises
**{A.n} simulations**. Results are also reported as an equivalent thermal conductivity
{SYM['k_eq']} = L<sub>eff</sub>/(A<sub>c</sub>\u00b7{SYM['r_th']}), evaluated at the
fixed operating geometry used throughout the study
(Q = {core.Q_WATT:.0f} W, A<sub>c</sub> = {core.A_CROSS:.3g} m\u00b2,
L<sub>eff</sub> = {core.L_EFF:.2f} m), so that a resistance can be read as a material-like
property and compared across configurations.
""", unsafe_allow_html=True)

st.markdown(
    f"""
**Surrogate modelling.** Rather than adopting one model on faith, six regression
families were trained on the same data and compared on equal terms: an artificial
neural network, random forest, gradient-boosted trees, support-vector regression,
Gaussian process regression and ridge polynomial regression. Because the dataset is
deliberately compact, generalisation was estimated by leave-one-out cross-validation
rather than a single hold-out split, which would have been dominated by the choice of
split. Inputs and outputs were standardised, and the pressure drop was modelled in a
logarithmic coordinate to respect its positivity and wide dynamic range. The selected
surrogate is a **{A.model_name}**{_r2_txt}, retained not only for accuracy but because a
Gaussian process returns a posterior standard deviation with every prediction: the model
reports where it is confident and where it is guessing, which is what makes the
uncertainty-aware features below possible.
""", unsafe_allow_html=True)

st.markdown(
    f"""
**Design optimisation and robustness.** With the surrogate in place, the design problem
becomes tractable: minimise {SYM['r_th']} subject to
{SYM['p_tot']} \u2264 a specified limit. Two independent solvers are used and
cross-checked - a multi-start sequential quadratic programming method that converges
onto the constraint boundary, and an exhaustive grid search that is globally exhaustive
on its lattice - so the reported optimum does not rest on a single algorithm. Sweeping
the limit traces a Pareto front whose slope is the constraint shadow price, quantifying
how much thermal resistance is bought per unit of additional allowable pressure drop.
Because a nominal optimum is only useful if it survives fabrication, tolerances on the
design variables are propagated by Monte-Carlo sampling through the surrogate to give
response distributions and a manufacturing yield, the probability that a fabricated unit
still satisfies the pressure-drop constraint.
""", unsafe_allow_html=True)

st.markdown(
    f"""
**Trustworthy use of a data-driven model.** A surrogate is only as good as the region it
was trained on, so every prediction in this application carries a domain-of-validity
assessment that combines three tests: whether the query lies inside the sampled bounds,
whether it lies inside the convex hull of the sampled designs, and where its predictive
standard deviation falls relative to the in-domain distribution. Predictions are labelled
high, moderate or low confidence accordingly, queries beyond the trained bounds are
refused rather than extrapolated, and predictive intervals accompany every reported
value. This is a deliberate methodological stance: the application is intended to support
engineering decisions, and a confident-looking number outside the data would be worse
than no number at all.
""", unsafe_allow_html=True)

st.markdown(
    f"""
**What this application provides.** The tool exposes the surrogate through eight modules:
single-point prediction with uncertainty and {SYM['k_eq']}; constrained optimisation with
the Pareto trade-off; manufacturing-tolerance and yield analysis; interactive
three-dimensional response surfaces with the constraint plane; a side-by-side performance
assessment of all trained models at a common design point; bulk evaluation of many designs
with export; a one-click PDF design report; and this attribution page. No model is trained
or re-fitted in the browser - every module loads the exact surrogate selected offline - so
the numbers shown here reproduce those reported in the associated study.
""", unsafe_allow_html=True)

st.caption("Built with Streamlit. The surrogate, scalers and dataset are produced by the "
           "authors' offline training pipeline and loaded here without retraining; the "
           "figures quoted above are read directly from the deployed artifacts.")
