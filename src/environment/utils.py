import json

class CrossProductActionSpace:
    def __init__(self, num_location: int):
        self.action_space = [i for i in range(num_location**2)]

        self.action_mapping = {}
        for loc1 in range(num_location):
            for loc2 in range(num_location):
                self.action_mapping[(loc1, loc2)] = loc1 * num_location + loc2

        self.inv_location_mapping = {v: k for k, v in self.action_mapping.items()}
        self.num_location = num_location

    def to_json(self):
        data = {"action_space": self.action_space,
                "action_mapping": {str(k): v for k, v in self.action_mapping.items()},
                "inv_location_mapping": self.inv_location_mapping,
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
        obj.inv_location_mapping = {int(k): v for k, v in action_mapping["inv_location_mapping"].items()}
        return obj

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.action_mapping[item]

        return self.inv_location_mapping[item]


if __name__ == "__main__":
    ca = CrossProductActionSpace.from_json("action_space.json")
    print(ca)