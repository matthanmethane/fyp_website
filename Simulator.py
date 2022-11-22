from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from SEIRNetwork import SEIRNetwork
from Simulation import Simulation


class Simulator:
    def __init__(
            self,
            population=4000000,
            numNodes=10000,
            network_config_path="seir_config.json",
            simulation_config_path="simulator_config.json"
    ) -> None:
        self.population = population
        self.numNodes = numNodes
        self.network_config_path = network_config_path
        self.simulation_config_path = simulation_config_path

        self.simulation = None
        self.SEIRModel = None
        self.model = None

        self._generate_seir_network()
        assert self.SEIRModel is not None
        assert self.model is not None

        self.intervention_history = []

    def _generate_seir_network(self):
        self.SEIRModel = SEIRNetwork(
            population=self.population,
            numNodes=self.numNodes,
            config_path=self.network_config_path
        )
        self.model = self.SEIRModel.model

    def set_checkpoint(self, new_checkpoint: Dict) -> None:
        """
        Set the new checkpoint.
        """
        pass

    def set_trace_together(
            self,
            start: float,
            end: float,
            tracing_lag: int,
            tracing_compliance_rate: float,
    ):
        """
        Set TraceTogether.
        """
        self.set_simulation_params(
            time=start,
            tracing_lag=tracing_lag,
            tracing_compliance=(
                    np.random.rand(self.numNodes) < tracing_compliance_rate)
        )
        self.set_simulation_params(
            time=end,
            tracing_lag=tracing_lag,
            tracing_compliance=(
                    np.random.rand(self.numNodes) < 0.00)
        )

    def set_imported_case(self, time: float, average_introductions_per_day: float):
        self.simulation.set_seir_network_params(
            time=time, average_introductions_per_day=average_introductions_per_day)

    # def set_social_distancing(self, time: float, global_rate: float):
    #     self.simulation.set_seir_network_params(time=time, p=global_rate)

    def set_social_distancing(self, start: float, end: float, global_rate: float):
        self.simulation.set_seir_network_params(time=start, p=global_rate)
        self.simulation.set_seir_network_params(time=end, p=0.5)

    # def set_social_gathering_limit(self, time: float, group_size: int):
    #     self.simulation.set_seir_network_params(
    #         time=time, G=self.SEIRModel.get_network("social_distancing", group_size))

    def set_social_gathering_limit(self, start: float, end: float, group_size: int):
        self.simulation.set_seir_network_params(
            time=start, G=self.SEIRModel.get_network("social_distancing", group_size))
        self.simulation.set_seir_network_params(time=end, G=self.SEIRModel.get_network("baseline"))

    def set_circuit_breaker(self, start: float, end: float):
        self.set_social_gathering_limit(start=start, end=end, group_size=0)
        self.set_social_distancing(start=start, end=end, global_rate=0.001)

    def set_seir_network_params(self, time: int, **kwargs) -> None:
        self.simulation.set_seir_network_params(time, **kwargs)

    def set_simulation_params(self, time, **kwargs) -> None:
        self.simulation.set_params(time, **kwargs)

    def generate_simulation(self, T=90):
        self.simulation = Simulation(
            model=self.model, numNodes=self.numNodes, T=T, config_path=self.simulation_config_path)

    def run(self):
        if not self.simulation:
            self.generate_simulation()
        while self.simulation.running:
            self.run_iteration()

    def run_iteration(self):
        if not self.simulation:
            self.generate_simulation()
        self.simulation.run()

# # pos = nx.drawing.layout.spring_layout(SEIR_network.G_household)
# # nx.draw_networkx(SEIR_network.G_household, pos,
# #                  with_labels=False, node_size=100)
# # plt.show()
# # print("NETOWRK PRINTED HAHAHAHAHAHAHAHA")
# simulator = Simulator()
# simulator.generate_simulation(T=150)


# # =================================================================================================
# # +++++ Feb 1st ~ May 1st +++++
# simulator.set_imported_case(time=1, average_introductions_per_day=0.1)
# simulator.set_trace_together(
#     time=1, tracing_lag=1, tracing_compliance_rate=0.0)
# simulator.set_social_distancing(time=68, global_rate=0.5)

# simulator.set_trace_together(
#     time=49, tracing_lag=1, tracing_compliance_rate=0.8)
# simulator.set_social_gathering_limit(time=57, group_size=10)
# simulator.set_imported_case(time=25, average_introductions_per_day=0.4)
# simulator.set_imported_case(time=40, average_introductions_per_day=1.0)
# simulator.set_social_distancing(time=68, global_rate=0.01)
# simulator.run()
# ax1 = plt.subplot(2, 1, 1)
# ax1.bar(simulator.simulation.numPosTseries.keys(),
#         simulator.simulation.numPosTseries.values())

# ax2 = plt.subplot(2, 1, 2)
# simulator.model.figure_infections(
#     ax=ax2,
#     plot_S=False,
#     plot_E=False,
#     plot_I_pre=False,
#     plot_I_sym='stacked',
#     plot_Q_sym='stacked',
#     plot_Q_pre=False,
#     plot_Q_asym=False,
#     plot_I_asym=False,
#     show=False,
#     combine_Q_infected=False,
#     plot_percentages=False,
#     scaled=True
# )

# # plt.title("Effect of Dynamic Scaling")
# #
# # ax1 = plt.subplot(2,2,1)
# # plt.title("Non-Scaled Non-S")
# # simulator.model.figure_infections(
# #     plot_S=False,
# #     plot_E="stacked",
# #     plot_I_pre="stacked",
# #     plot_I_sym="stacked",
# #     plot_I_asym="stacked",
# #     plot_H="stacked",
# #     plot_R="stacked",
# #     plot_F="stacked",
# #     plot_Q_E="stacked",
# #     plot_Q_pre="stacked",
# #     plot_Q_sym="stacked",
# #     plot_Q_asym="stacked",
# #     plot_Q_S=False,
# #     plot_Q_R="stacked",
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     show=False,
# #     scaled=False,
# #     ax=ax1
# # )
# # ax2 = plt.subplot(2,2,2)
# # plt.title("Scaled Non-S")
# # simulator.model.figure_infections(
# #     plot_S=False,
# #     plot_E="stacked",
# #     plot_I_pre="stacked",
# #     plot_I_sym="stacked",
# #     plot_I_asym="stacked",
# #     plot_H="stacked",
# #     plot_R="stacked",
# #     plot_F="stacked",
# #     plot_Q_E="stacked",
# #     plot_Q_pre="stacked",
# #     plot_Q_sym="stacked",
# #     plot_Q_asym="stacked",
# #     plot_Q_S=False,
# #     plot_Q_R="stacked",
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     show=False,
# #     scaled=True,
# #     ax=ax2
# # )
# #
# # ax3 = plt.subplot(2,2,3)
# # plt.title("Non-Scaled SEIR")
# # simulator.model.figure_basic(
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     plot_S='stacked',
# #     plot_Q_S='stacked',
# #     show=False,
# #     scaled=False,
# #     ax=ax3
# # )
# # ax4 = plt.subplot(2,2,4)
# # plt.title("Scaled SEIR")
# # simulator.model.figure_basic(
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     plot_S='stacked',
# #     plot_Q_S='stacked',
# #     plot_Q_R='line',
# #     show=False,
# #     scaled=True,
# #     ax=ax4
# # )
# #
# #
# # plt.suptitle("Effect of Dynamic Scaling")
# plt.show()
