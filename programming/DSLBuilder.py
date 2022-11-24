import yaml
import programming.dsl_defaults as defaults


class DSLBuilder:
    def __init__(self, yaml_filename=None):
        self.yaml_filename = yaml_filename
        self.yaml_dsl = {}
        if yaml_filename is not None:
            with open(yaml_filename, "r") as file:
                self.yaml_dsl = yaml.load(file)

    def _build_pipeline_step(
        self,
        step_type,
        step_name,
    ):
        step = {
            "name": step_name,
            "description": defaults.description[step_type],
            "type": step_type,
            "operation": defaults.operation[step_type],
        }

        return step

    def build_pipeline(self, pipeline_id, symbols_class, text_labels):
        if pipeline_id not in self.yaml_dsl.keys():
            self.yaml_dsl[pipeline_id] = {}
        self.yaml_dsl[pipeline_id]["name"] = "Pipeline Name"
        self.yaml_dsl[pipeline_id]["type"] = "Pipeline"
        self.yaml_dsl[pipeline_id]["steps"] = {}
        for i, (step_type, step_name) in enumerate(zip(symbols_class, text_labels)):
            step_id = "step_" + str(i)
            step = self._build_pipeline_step(step_type=step_type, step_name=step_name)
            self.yaml_dsl[pipeline_id]["steps"][step_id] = step
        return self.yaml_dsl[pipeline_id]

    def save_yaml(self, filename=None):
        if filename is None:
            filename = self.yaml_filename

        with open(filename, "w") as file:
            yaml.Dumper.ignore_aliases = lambda *args: True
            yaml.dump(
                self.yaml_dsl,
                file,
                default_flow_style=False,
                allow_unicode=False,
                sort_keys=False,
            )
