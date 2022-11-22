from collections import defaultdict, Counter
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from Simulator import Simulator

# ==================================================================
# Set Page Configuration
# ==================================================================
st.set_page_config(
    page_title="COVID-19 Singapore Simulator",
    page_icon='🇸🇬',
    layout='centered'
)
st.title("COVID-19 Simulator")

# ==================================================================
# State Management Initialization
# ==================================================================
if 'simulator' not in st.session_state:
    st.session_state.simulator = None

if 'simulator_configs' not in st.session_state:
    st.session_state.simulator_configs = defaultdict(list)

if 'simulate_btn' not in st.session_state:
    st.session_state.simulate_btn = False

if 'simulation_time' not in st.session_state:
    st.session_state.simulation_time = 120


# ==================================================================
# Helper Functions
# ==================================================================

def generate_simulator():
    if not st.session_state.simulator:
        st.session_state.simulator = Simulator()


@st.experimental_singleton(suppress_st_warning=True)
def plot_distribution(network_type, social_gathering_limit: Optional[int] = None):
    if social_gathering_limit:
        graph = st.session_state.simulator.SEIRModel.get_network(network_type, social_gathering_limit)
    else:
        graph = st.session_state.simulator.SEIRModel.get_network(network_type)
    nodeDegrees = [d[1] for d in graph.degree()]
    meanDegree = np.mean(nodeDegrees)
    fig, ax = plt.subplots()
    ax.hist(nodeDegrees, bins=range(max(nodeDegrees)), alpha=0.75, color='tab:blue',
            label=('mean degree = %.1f' % meanDegree))
    plt.xlim(0, max(nodeDegrees))
    ax.legend(loc='upper right')
    return fig


def add_restriction(restriction_type, days):
    start_day, end_day = days
    if days not in st.session_state.simulator_configs[restriction_type]:
        st.session_state.simulator_configs[restriction_type].append([start_day, end_day])
        st.session_state.simulator_configs[restriction_type].sort()
        temp = st.session_state.simulator_configs[restriction_type][1:]
        start1, end1 = st.session_state.simulator_configs[restriction_type][0]
        st.session_state.simulator_configs[restriction_type] = []
        for start2, end2 in temp:
            if start2 <= end1:
                end1 = max(end1, end2)
            else:
                st.session_state.simulator_configs[restriction_type].append([start1, end1])
                start1, end1 = start2, end2
        st.session_state.simulator_configs[restriction_type].append([start1, end1])


def reset_restriction():
    st.session_state.simulator_configs = defaultdict(list)


def apply_restriction():
    for restriction in st.session_state.simulator_configs:
        for start, end in st.session_state.simulator_configs[restriction]:
            if restriction == "Trace Together":
                st.session_state.simulator.set_trace_together(
                    start=start,
                    end=end,
                    tracing_lag=1,
                    tracing_compliance_rate=0.9
                )
            elif restriction == "Social Distancing":
                st.session_state.simulator.set_social_distancing(
                    start=start,
                    end=end,
                    global_rate=0.2
                )
            elif restriction == "Circuit Breaker":
                st.session_state.simulator.set_circuit_breaker(start=start, end=end)
            elif restriction == "Social Gathering Limit":
                st.session_state.simulator.set_social_gathering_limit(
                    start=start,
                    end=end,
                    group_size=5
                )


# ==================================================================
# Main Snippet
# ==================================================================
with st.spinner("Initializing Simulator"):
    generate_simulator()

# ==================================================================
# Network Degree Distribution
# ==================================================================
baseline_tab, quarantine_tab, household_tab, social_gathering_tab = st.tabs(
    ["Baseline", "Quarantine", "Household", "Social Gathering Limit"])
with baseline_tab:
    G = st.session_state.simulator.SEIRModel.get_network('baseline')
    st.text(G)
    st.text("Representing 4,000,000 number of population")
    degrees = [G.degree(n) for n in G.nodes()]
    st.subheader("Mean Degree (Contact) Per Person): " + str(np.mean(degrees)))
    degree_distribution = Counter(degrees)
    df = pd.DataFrame()
    df['Degree'] = degree_distribution.keys()
    df['Frequency'] = degree_distribution.values()
    st.bar_chart(df, x='Degree')
with quarantine_tab:
    G = st.session_state.simulator.SEIRModel.get_network('quarantine')
    st.text(G)
    st.text("Representing 4,000,000 number of population")
    degrees = [G.degree(n) for n in G.nodes()]
    st.subheader("Mean Degree (Contact) Per Person): " + str(np.mean(degrees)))
    degree_distribution = Counter(degrees)
    df = pd.DataFrame()
    df['Degree'] = degree_distribution.keys()
    df['Frequency'] = degree_distribution.values()
    st.bar_chart(df, x='Degree')
with household_tab:
    G = st.session_state.simulator.SEIRModel.get_network('household')
    st.text(G)
    st.text("Representing 4,000,000 number of population")
    degrees = [G.degree(n) for n in G.nodes()]
    st.subheader("Mean Degree (Contact) Per Person): " + str(np.mean(degrees)))
    degree_distribution = Counter(degrees)
    df = pd.DataFrame()
    df['Degree'] = degree_distribution.keys()
    df['Frequency'] = degree_distribution.values()
    st.bar_chart(df, x='Degree')
with social_gathering_tab:
    social_gathering_val = social_gathering_tab.slider('Gathering Size Limit', min_value=0, max_value=50, value=0)
    if social_gathering_val:
        G = st.session_state.simulator.SEIRModel.get_network('social_gathering', social_gathering_val)
        st.text(G)
        st.text("Representing 4,000,000 number of population")
        degrees = [G.degree(n) for n in G.nodes()]
        st.subheader("Mean Degree (Contact) Per Person): " + str(np.mean(degrees)))
        degree_distribution = Counter(degrees)
        df = pd.DataFrame()
        df['Degree'] = degree_distribution.keys()
        df['Frequency'] = degree_distribution.values()
        st.bar_chart(df, x='Degree')

# ==================================================================
# Main Snippet
# ==================================================================
st.title("Virus Configuration")
st.text("Default: Average Parameters of COVID-19")

virus_col1, virus_col2 = st.columns(2)
with virus_col1:
    st.number_input(
        "Latent Period",
        min_value=0.00,
        max_value=50.00,
        value=5.20,
        format="%f",
        key="latent_period"
    )
    st.number_input(
        "Pre-Symptomatic Period",
        min_value=0.00,
        max_value=20.00,
        value=5.00,
        format="%f",
        key="presymptomatic_period"
    )
    st.number_input(
        "Asymptomatic Probability",
        min_value=0.00,
        max_value=1.00,
        value=0.5,
        format="%f",
        key="asymp_prob"
    )
with virus_col2:
    st.number_input(
        "Infectious Period",
        min_value=0.00,
        max_value=50.00,
        value=10.00,
        format="%f",
        key="infectious_period"
    )
    st.number_input(
        "Basic Reproductive Rate (R0)",
        min_value=0.00,
        max_value=50.00,
        value=2.5,
        format="%f",
        key="r2"
    )
    st.number_input(
        "Average Introductions Per Day",
        min_value=0.00,
        max_value=1.00,
        value=1.00,
        format="%f",
        key="average_introductions_per_day	"
    )

with st.expander("Adjust initial number of infections"):
    initial_num1, initial_num2, initial_num3 = st.columns(3)
    with initial_num1:
        st.number_input(
            "Number of Exposed Individual",
            min_value=0,
            max_value=4000000,
            key="numE"
        )
    with initial_num2:
        st.number_input(
            "Number of Infected Individual",
            min_value=0,
            max_value=4000000,
            key="numI"
        )
    with initial_num3:
        st.number_input(
            "Number of Recovered Individual",
            min_value=0,
            max_value=4000000,
            key="numR"
        )

trans_by_age = st.radio(
    "Transmissibility by Age Group",
    ('Use Default Setting', 'Identical For All Age Groups', 'Set Custom Rates'))

if trans_by_age == "Use Default Setting":
    st.text("Young: 0.3")
    st.text("Teenager: 1.2")
    st.text("Adult: 1.0")
    st.text("Old: 0.8")
elif trans_by_age == "Set Custom Rates":
    age_column1, age_column2, age_column3, age_column4 = st.columns(4)
    with age_column1:
        st.text("Young")
        st.number_input("0 - 4", min_value=0.00, max_value=2.00, value=0.3, key="0_4_beta")
        st.number_input("5 - 9", min_value=0.00, max_value=2.00, value=0.3, key="5_9_beta")
    with age_column2:
        st.text("Teenager")
        st.number_input("10 - 14", min_value=0.00, max_value=2.00, value=1.2, key="10_14_beta")
        st.number_input("15 - 19", min_value=0.00, max_value=2.00, value=1.2, key="15_19_beta")
    with age_column3:
        st.text("Adult")
        st.number_input("20 - 24", min_value=0.00, max_value=2.00, value=1.0, key="20_24_beta")
        st.number_input("25 - 29", min_value=0.00, max_value=2.00, value=1.0, key="25_29_beta")
        st.number_input("30 - 34", min_value=0.00, max_value=2.00, value=1.0, key="30_34_beta")
        st.number_input("35 - 39", min_value=0.00, max_value=2.00, value=1.0, key="35_39_beta")
        st.number_input("40 - 44", min_value=0.00, max_value=2.00, value=1.0, key="40_44_beta")
        st.number_input("45 - 49", min_value=0.00, max_value=2.00, value=1.0, key="45_49_beta")
        st.number_input("50 - 54", min_value=0.00, max_value=2.00, value=1.0, key="50_54_beta")
        st.number_input("55 - 59", min_value=0.00, max_value=2.00, value=1.0, key="55_59_beta")
        st.number_input("60 - 64", min_value=0.00, max_value=2.00, value=1.0, key="60_64_beta")
    with age_column4:
        st.text("Old")
        st.number_input("65 - 69", min_value=0.00, max_value=2.00, value=0.8, key="65_69_beta")
        st.number_input("70 - 74", min_value=0.00, max_value=2.00, value=0.8, key="70_74_beta")
        st.number_input("75 - 79", min_value=0.00, max_value=2.00, value=0.8, key="75_79_beta")
        st.number_input("80 - 84", min_value=0.00, max_value=2.00, value=0.8, key="80_84_beta")
        st.number_input("85 - 89", min_value=0.00, max_value=2.00, value=0.8, key="85_89_beta")
        st.number_input("90 Above", min_value=0.00, max_value=2.00, value=0.8, key="90_beta")

# ==================================================================
# Simulator Configuration
# ==================================================================
st.title("Simulator Configuration")
st.slider(
    "Time of Simulation",
    min_value=0,
    max_value=365,
    key="simulation_time"
)

st.title("Government Restriction")
st.selectbox(
    "Choose Restriction Type",
    options=["Trace Together", "Social Distancing", "Social Gathering Limit", "Circuit Breaker", ],
    key='restriction_type'
)

if st.session_state.restriction_type == "Social Gathering Limit":
    st.slider(
        "Social Gathering Limit Size",
        min_value=0,
        max_value=50,
        value=10,
        key="gathering_limit_size"
    )
elif st.session_state.restriction_type == "Social Distancing":
    st.slider(
        "Rate of Social Distancing",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        format="%f",
        key="social_distance_rate"
    )

st.slider(
    "Time (Start -> End)",
    min_value=0,
    max_value=st.session_state.simulation_time,
    value=(0, 120),
    key="restriction_time"
)
st.button(
    "Add Restriction",
    on_click=add_restriction,
    args=(st.session_state.restriction_type, st.session_state.restriction_time)
)
st.button(
    "Reset Restrictions",
    on_click=reset_restriction
)
st.write(st.session_state.simulator_configs)
st.session_state.simulate_btn = st.button("Simulate")
st.write(st.session_state.simulate_btn)

# ==================================================================
# Simulation Result
# ==================================================================
if st.session_state.simulate_btn:
    st.write(st.session_state.simulator)
    with st.spinner("Simulating..."):
        st.write(st.session_state.simulation_time)
        st.session_state.simulator.generate_simulation(st.session_state.simulation_time)
        apply_restriction()
        st.session_state.simulator.run()

    pos_case_tab, infected_tab, seir_tab, custom_tab = st.tabs(
        ["Positive Case", "Infected Case", "SEIR", "Custom"])
    S_series = st.session_state.simulator.model.numS * st.session_state.simulator.model.scale_series
    E_series = st.session_state.simulator.model.numE * st.session_state.simulator.model.scale_series
    I_pre_series = st.session_state.simulator.model.numI_pre * st.session_state.simulator.model.scale_series
    I_sym_series = st.session_state.simulator.model.numI_sym * st.session_state.simulator.model.scale_series
    I_asym_series = st.session_state.simulator.model.numI_asym * st.session_state.simulator.model.scale_series
    R_series = st.session_state.simulator.model.numR * st.session_state.simulator.model.scale_series
    Q_S_series = st.session_state.simulator.model.numQ_S * st.session_state.simulator.model.scale_series
    Q_E_series = st.session_state.simulator.model.numQ_E * st.session_state.simulator.model.scale_series
    Q_pre_series = st.session_state.simulator.model.numQ_pre * st.session_state.simulator.model.scale_series
    Q_sym_series = st.session_state.simulator.model.numQ_sym * st.session_state.simulator.model.scale_series
    Q_asym_series = st.session_state.simulator.model.numQ_asym * st.session_state.simulator.model.scale_series
    Q_R_series = st.session_state.simulator.model.numQ_R * st.session_state.simulator.model.scale_series

    with pos_case_tab:
        df = pd.DataFrame(st.session_state.simulator.simulation.numPosTseries.values(), columns=["Cases"])
        st.bar_chart(data=df)
    with infected_tab:
        df = pd.DataFrame()
        df[
            'Infected (Total)'] = I_pre_series + I_sym_series + I_asym_series + Q_pre_series + Q_sym_series + Q_asym_series
        df['Infected & Quarantined (Pre-symptomatic)'] = Q_pre_series
        df['Infected & Quarantined (Symptomatic)'] = Q_sym_series
        df['Infected & Quarantined (Asymptomatic)'] = Q_asym_series
        df['Infected (Pre-symptomatic)'] = I_pre_series
        df['Infected (Symptomatic)'] = I_sym_series
        df['Infected (Asymptomatic)'] = I_asym_series
        df['time'] = st.session_state.simulator.model.tseries
        st.area_chart(data=df, x='time')
    with seir_tab:
        df = pd.DataFrame()
        df['Susceptible (Total)'] = st.session_state.simulator.population - (E_series + Q_E_series) - (
                I_pre_series + I_sym_series + I_asym_series + Q_pre_series + Q_sym_series + Q_asym_series) - (
                                            R_series + Q_R_series)
        df['Exposed (Total)'] = E_series + Q_E_series
        df[
            'Infected (Total)'] = I_pre_series + I_sym_series + I_asym_series + Q_pre_series + Q_sym_series + Q_asym_series
        df['Recovered (Total)'] = R_series + Q_R_series
        df['time'] = st.session_state.simulator.model.tseries
        st.area_chart(data=df, x='time')
