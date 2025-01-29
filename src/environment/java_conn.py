from contextlib import contextmanager
from typing import List, Union, Any, Dict
import jpype
import jpype.imports
from dataclasses import dataclass
from typing import Tuple, List, Dict

from src.utils import initialize_logger

@dataclass
class FluidityStepResult:
    distance_latencies: Dict[str, Dict[str, float]]
    request_quantity: Dict[str, float]
    mean_delay: float
    coordinator: str
    active_locations: List[str]
    passive_locations: List[str]
    finished: bool = False


@contextmanager
def jvm_context(classpath: List[str], jvm_options: List[str] = []):
    """
    A context manager to manage the lifecycle of the JVM.
    :param classpath: List of paths or a single path string to the class files or JARs.
    """
    try:
        # Start JVM if not already started
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), *jvm_options, classpath=classpath)
        yield
    finally:
        if jpype.isJVMStarted():
            jpype.shutdownJVM()

logger = initialize_logger()

class JavaSimulator:
    def __init__(self, jar_path: str, jvm_options: List[str],
                 configuration_directory_simulator: str,
                 node_identifier: str = "server0",):
        self.jar_path = jar_path

        # Fully qualified Java class names
        # P is for path
        self.FluiditySimulator = "de.optscore.simulator.FluiditySimulator"
        self.FluiditySimulationExecution = "de.optscore.simulation.fluidity.FluiditySimulationExecution"
        self.FluiditySimulationExecutionFactory = "de.optscore.simulation.fluidity.FluiditySimulationExecutionFactory"
        self.FluiditySimulationConfiguration = "de.optscore.simulation.fluidity.configuration.FluiditySimulationConfiguration"
        self.FluidityStepAction = "de.optscore.simulation.fluidity.step.FluidityStepAction"
        self.FluiditySimulation = "de.optscore.simulation.fluidity.FluiditySimulation"
        self.DefaultOptimizationInstructions = "bftsmart.location.management.exploration.DefaultOptimizationInstructions"
        self.ArrayList = "java.util.ArrayList"
        self.SwapActiveKeepPassiveP = "bftsmart.location.management.exploration.instructions.SwapActiveKeepPassive"
        self.SwapActiveP = "bftsmart.location.management.exploration.instructions.SwapActive"
        self.LocatedNodeIdentifierP = "bftsmart.identity.LocatedNodeIdentifier"
        self.LocatedNodeP = "bftsmart/location/LocatedNode"
        self.EnsureCoordinatorP = "bftsmart.location.management.exploration.instructions.EnsureCoordinator"

        self.configuration_directory_simulator = configuration_directory_simulator
        self.node_identifier = node_identifier
        self.jvm_options = jvm_options

        self.classpath = [self.jar_path]

        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), *self.jvm_options, classpath=self.classpath)

        # Load Java classes
        self._initialize_objects()
        self.internal_step = 0


    def _initialize_objects(self):
        self.Simulator = jpype.JClass(self.FluiditySimulator)
        self.Simulation = jpype.JClass(self.FluiditySimulation)
        self.ExecutionFactory = jpype.JClass(self.FluiditySimulationExecutionFactory)
        self.Execution = jpype.JClass(self.FluiditySimulationExecution)
        self.Config = jpype.JClass(self.FluiditySimulationConfiguration)
        self.StepAction = jpype.JClass(self.FluidityStepAction)
        self.OptimizationInstructions = jpype.JClass(self.DefaultOptimizationInstructions)
        self.JavaArrayList = jpype.JClass(self.ArrayList)
        self.Paths = jpype.JClass("java.nio.file.Paths")
        self.SwapActiveKeepPassive = jpype.JClass(self.SwapActiveKeepPassiveP)
        self.SwapActive = jpype.JClass(self.SwapActiveP)
        self.LocatedNodeIdentifier = jpype.JClass(self.LocatedNodeIdentifierP)
        self.LocatedNode = jpype.JClass(self.LocatedNodeP)
        self.EnsureCoordinator = jpype.JClass(self.EnsureCoordinatorP)

        self.config_dir = self.Paths.get(self.configuration_directory_simulator + "/" + self.node_identifier)

        self.simulator = self.Simulator(self.configuration_directory_simulator, self.node_identifier)
        self.config = self.Config(self.config_dir, self.node_identifier)
        self.simulation = self.Simulation(self.config.getSelf(), self.config.getXmrConfigurationDirectory(),
                                          self.config)
        self.execution = self.ExecutionFactory().create(self.config)
        self.coordinator = self.simulation.getState().getView().getCoordinator()
        self.passive_nodes = self.simulation.getState().getView().getPassiveNodes()


    def step(self, action: Tuple[int, int]) -> FluidityStepResult:
        step_action = self._convert_action(action[0], action[1])
        step_result = self.execution.executeStep(step_action)
        client_latencies = step_result.getState().getSystemLatencies().getReplicaClientLatencies()
        replica_latencies = step_result.getState().getSystemLatencies().getReplicaLatencies()
        self.internal_step += 1

        active_locations = [x.getLocation().identify() for x in step_result.getState().getView().getActiveView().getNodes()]
        passive_locations = [x.getLocation().identify() for x in step_result.getState().getView().getPassiveView().getNodes()]
        mean_latency = step_result.getState().getAverageCalculation().getAverageLatency()

        self.coordinator = step_result.getState().getView().getCoordinator()
        self.passive_nodes = step_result.getState().getView().getPassiveNodes()

        distance_latencies, request_quantity = self._convert_step_result(client_latencies, replica_latencies,
                                                                         step_result.getState().getSystemLatencies().getRequestCounts())
        if self.execution.getWorkload().getNumberOfSteps() == (self.internal_step):
            self.internal_step = 0
            result = FluidityStepResult(distance_latencies=distance_latencies,
                                      mean_delay=mean_latency,
                                      request_quantity=request_quantity,
                                      active_locations=active_locations,
                                      coordinator=self.coordinator.getLocation().identify(),
                                      passive_locations=passive_locations,
                                      finished=True)
            self._initialize_objects()
            return result

        return FluidityStepResult(distance_latencies=distance_latencies,
                                  mean_delay=mean_latency,
                                  request_quantity=request_quantity,
                                  active_locations=active_locations,
                                  coordinator=self.coordinator.getLocation().identify(),
                                  passive_locations=passive_locations,
                                  finished=False)

    def _convert_step_result(self, client_latencies, replica_latencies, request_count):
        client_nodes = [x for x in request_count.getNodes()]
        client_request_quant = {x.identify() : request_count.getCount(x) for x in client_nodes}
        parsed_client_lats = {entry.getKey().getLocation().identify(): entry.getValue() for entry in client_latencies.entrySet()}
        parsed_replica_lats = {entry.getKey().getLocation().identify(): entry.getValue() for entry in replica_latencies.entrySet()}
        final_parse_clients = {}
        for k, v in parsed_client_lats.items():
            final_parse_clients[k] = {entry.getKey().identify(): entry.getValue() for entry in v.entrySet()}

        final_parse_replicas = {}
        for k, v in parsed_replica_lats.items():
            final_parse_replicas[k] = {str(entry.getKey().identify()).split(" ")[3].strip(","): entry.getValue()
                                       for entry in v.entrySet()}

        def merge_nested_dicts(dict1, dict2):
            merged = {}
            for key in set(dict1.keys()).union(set(dict2.keys())):
                if key in dict1 and key in dict2:
                    if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        merged[key] = merge_nested_dicts(dict1[key], dict2[key])
                    else:
                        merged[key] = dict2[key]
                elif key in dict1:
                    merged[key] = dict1[key]
                else:
                    merged[key] = dict2[key]
            return merged

        return merge_nested_dicts(final_parse_clients, final_parse_replicas), client_request_quant


    def _convert_action(self, add_action: int, remove_action: int):
        if add_action == remove_action:
            logger.debug("Noop")
            return self.create_noop()
        else:
            return self._create_swap_active(add_action, remove_action)

    def create_noop(self):
        return self.StepAction(self.OptimizationInstructions(self.JavaArrayList()))

    def _create_swap_active(self, to_add: int, to_remove: int):
        all_nodes = self.simulation.getState().getView().getAvailableNodes()

        replicaIDAdd = all_nodes.get(to_add)
        replicaIDRemove = all_nodes.get(to_remove)
        locatedAddNode = self.LocatedNode(replicaIDAdd, replicaIDAdd.getLocation())
        locatedRemoveNode = self.LocatedNode(replicaIDRemove, replicaIDRemove.getLocation())

        if any([x.equals(replicaIDAdd) for x in  self.passive_nodes]):
            #logger.debug("Need to swap active and passive")
            #logger.debug(f"SwapActive: {replicaIDAdd} -> {replicaIDRemove}")
            swapActive = self.SwapActive(locatedAddNode, locatedRemoveNode)
        else:
            #logger.debug("Normal swap")
            #logger.debug(f"SwapActiveKeepPassive: {replicaIDAdd} -> {replicaIDRemove}")
            swapActive = self.SwapActiveKeepPassive(locatedAddNode, locatedRemoveNode)

        action_list = self.JavaArrayList()
        action_list.add(swapActive)

        if (self.coordinator.equals(replicaIDRemove)):
            coordinator = self.EnsureCoordinator(replicaIDAdd)
            action_list.add(coordinator)

        fluidity_action = self.StepAction(self.OptimizationInstructions(action_list))

        return fluidity_action


if __name__ == "__main__":
    conn = JavaSimulator(jvm_options = [
    '-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
        jar_path="/home/lukas/Projects/emusphere/simulator-xmr/target/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        configuration_directory_simulator="/home/lukas/flusim/simurun/"

    )

    conn.step((4, 0))
    conn.step((0, 6))