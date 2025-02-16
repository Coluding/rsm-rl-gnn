import json
from typing import List

class CrossProductActionSpace:
    def __init__(self, num_location: int):
        self.action_space = [i for i in range(num_location * (num_location - 1) + 1)]

        self.action_mapping = {}
        counter = 0
        for loc1 in range(num_location):
            for loc2 in range(num_location):
                if loc1 != loc2:
                    self.action_mapping[(loc1, loc2)] = counter
                    counter += 1

        self.action_mapping[(-1, -1)] = len(self.action_space) - 1

        self.inv_action_mapping = {v: k for k, v in self.action_mapping.items()}
        self.num_location = num_location

    def to_json(self):
        data = {"action_space": self.action_space,
                "action_mapping": {str(k): v for k, v in self.action_mapping.items()},
                "inv_action_mapping": self.inv_action_mapping,
                "num_location": self.num_location}

        with open("action_space.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def from_json(cls, file):
        with open(file, "r") as f:
            action_mapping = json.load(f)

        obj = cls(action_mapping["num_location"])
        obj.action_space = action_mapping["action_space"]
        obj.action_mapping = {eval(k): v for k, v in action_mapping["action_mapping"].items()}
        obj.inv_location_mapping = {int(k): v for k, v in action_mapping["inv_action_mapping"].items()}
        return obj

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.action_mapping[item]

        return self.inv_action_mapping[item]

    def build_add_action_mask(self, addable_locations: List[int]):
        mask = []
        for key in sorted(self.inv_action_mapping.keys()):
            if self.inv_action_mapping[key][0] in addable_locations or self.inv_action_mapping[key][0] == -1:
                mask.append(0)
            else:
                mask.append(-float("inf"))
        return mask

    def build_remove_action_mask(self, removable_locations: List[int]):
        mask = []
        for key in sorted(self.inv_action_mapping.keys()):
            if self.inv_action_mapping[key][1] in removable_locations or self.inv_action_mapping[key][0] == -1:
                mask.append(0)
            else:
                mask.append(-float("inf"))
        return mask


if __name__ == "__main__":
    ca = CrossProductActionSpace(8)
    ca.to_json()
    print(ca)