from powerowl.graph.model_node import ModelNode


class OrganizationalUnit(ModelNode):
    def __str__(self):
        return f'"OU: {self.name}"'
