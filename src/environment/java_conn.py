from contextlib import contextmanager
from typing import List
import jpype
import jpype.imports


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


class JavaSimulator:
    def __init__(self, jar_path: str, jvm_options: List[str],
                 configuration_directory_simulator: str,
                 node_identifier: str = "server0",):
        self.jar_path = jar_path

        # Fully qualified Java class names
        self.FluiditySimulator = "de.optscore.simulator.FluiditySimulator"
        self.FluiditySimulationExecution = "de.optscore.simulation.fluidity.FluiditySimulationExecution"
        self.FluiditySimulationExecutionFactory = "de.optscore.simulation.fluidity.FluiditySimulationExecutionFactory"
        self.FluiditySimulationConfiguration = "de.optscore.simulation.fluidity.configuration.FluiditySimulationConfiguration"
        self.FluidityStepAction = "de.optscore.simulation.fluidity.step.FluidityStepAction"
        self.DefaultOptimizationInstructions = "bftsmart.location.management.exploration.DefaultOptimizationInstructions"
        self.ArrayList = "java.util.ArrayList"

        self.jvm_options = jvm_options

        self.classpath = [self.jar_path]

        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), *self.jvm_options, classpath=self.classpath)

        # Load Java classes
        self.Simulator = jpype.JClass(self.FluiditySimulator)
        self.ExecutionFactory = jpype.JClass(self.FluiditySimulationExecutionFactory)
        self.Execution = jpype.JClass(self.FluiditySimulationExecution)
        self.Config = jpype.JClass(self.FluiditySimulationConfiguration)
        self.StepAction = jpype.JClass(self.FluidityStepAction)
        self.OptimizationInstructions = jpype.JClass(self.DefaultOptimizationInstructions)
        self.JavaArrayList = jpype.JClass(self.ArrayList)
        self.Paths = jpype.JClass("java.nio.file.Paths")

        self.config_dir = self.Paths.get(configuration_directory_simulator + "/" + node_identifier)
        self.node_identifier = node_identifier

        self.simulator = self.Simulator(configuration_directory_simulator, self.node_identifier)
        self.config = self.Config(self.config_dir, self.node_identifier)
        self.execution = self.ExecutionFactory().create(self.config)


    def step(self, action: int):
        instruction = self._convert_action(action)
        step_action = self.StepAction(instruction)
        step_result = self.execution.executeStep(step_action)
        client_latencies = step_result.getSystemLatencies().getReplicaClientLatencies()
        replica_latencies = step_result.getSystemLatencies().getReplicaLatencies()

        return self._convert_step_result(client_latencies, replica_latencies)

    def _convert_step_result(self, client_latencies, replica_latencies):
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

        return merge_nested_dicts(final_parse_clients, final_parse_replicas)


    def _convert_action(self, action: int):
        return self.OptimizationInstructions(self.JavaArrayList())


if __name__ == "__main__":
    conn = JavaSimulator(jvm_options = [
    '-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
        jar_path="/home/lukas/Projects/emusphere/simulator-xmr/target/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        configuration_directory_simulator="/home/lukas/flusim/simurun/"

    )

    conn.step(1)