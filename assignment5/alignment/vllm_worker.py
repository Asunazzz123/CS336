"""Optional read-only worker RPC used to verify real cross-GPU weight transfer."""
from __future__ import annotations
import json


class WeightSyncProbe:
    def weight_sync_probe(self, names: list[str] | str) -> dict:
        if isinstance(names, str):
            names = json.loads(names)
        if not 1 <= len(names) <= 8:
            raise ValueError("Probe one to eight named parameters")
        model = self.model_runner.get_model()
        parameters = dict(model.named_parameters(remove_duplicate=False))
        result = {}
        for name in names:
            parameter = parameters[name]
            values = []
            for index in range(min(4, parameter.numel())):
                remainder, coordinates = index, []
                for dimension in reversed(parameter.shape):
                    coordinates.append(remainder % dimension)
                    remainder //= dimension
                value = parameter[tuple(reversed(coordinates))]
                values.append(float(value.detach().float().cpu()))
            result[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                            "values": values}
        return result
