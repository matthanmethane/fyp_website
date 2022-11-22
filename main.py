import streamlit as st

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
# Helper Functions
# ==================================================================
def set_true(item):
    st.session_state[item] = True


def set_value(item, value):
    st.session_state[item] = value


def prettify_param(param_name):
    return ' '.join([x.capitalize() for x in param_name.split('_')])


# ==================================================================
# Network Generator Page
# ==================================================================
st.title("Network and Virus Configuration")

# ==================================================================
# CallBack Functions
# ==================================================================


# ==================================================================
# State Management Initialization
# ==================================================================
network_param_list = [
    ('latent_period', 0, 20, 10, 'int'),
    ('infectious_period', 0, 20, 10, 'int'),
    ('presymptomatic_period', 0, 20, 10, 'int'),
    ('r0', 0.0, 50.0, 2.5, 'float'),
    ('asymp_prob', 0.0, 1.0, 0.5, 'float'),
    ('global_infectiotion_rate', 0.000, 1.0000, 0.5, 'float'),
    ('global_infectiotion_rate_q', 0.0000, 1.0000, 0.001, 'float'),
    ('quarantine_days', 0, 21, 14, 'int')
]

if 'use_default_param_btn' not in st.session_state:
    st.session_state.use_default_param_btn = False
if 'generate_network_btn' not in st.session_state:
    st.session_state.generate_network_btn = False

if st.session_state['use_default_param_btn']:
    st.session_state['use_default_param_btn'] = False
    for param, min_val, max_val, default_val, data_type in network_param_list:
        st.session_state[param] = default_val

for param, min_val, max_val, default_val, data_type in network_param_list:
    if param not in st.session_state:
        st.session_state[param] = default_val
    st.number_input(
        label=prettify_param(param),
        min_value=min_val,
        max_value=max_val,
        value=st.session_state[param],
        format="%f" if data_type == 'float' else '%d',
        key=param
    )

# ==================================================================
# Main Snippet
# ==================================================================

st.button("Use Default Parameters", on_click=set_true, args=('use_default_param_btn',))

# ==================================================================
# Simulator Generator Page
# ==================================================================
# ==================================================================
# CallBack Functions
# ==================================================================


# ==================================================================
# State Management Initialization
# ==================================================================
st.button("Generate Network", on_click=set_true, args=('generate_network_btn',))

if st.session_state['generate_network_btn']:
    st.session_state['generate_network_btn'] = False
    st.title("Network and Virus Configuration")
    for param, min_val, max_val, default_val, data_type in network_param_list:
        st.write(st.session_state[param])
    simulator = Simulator()

# ==================================================================
# Simulation Viewer Page
# ==================================================================
