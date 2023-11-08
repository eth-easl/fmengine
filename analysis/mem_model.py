from dataclasses import dataclass, asdict
from tabulate import tabulate


@dataclass
class MemModel:
    V = 32000
    h = 4096
    f = 11008
    r = 8
    L = 32
    a = 32
    b = 8
    s = 1024

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def print_config(self):
        print(f"V={self.V}, h={self.h}, f={self.f}, r={self.r}, L={self.L}, a={self.a}, b={self.b}, s={self.s}")

    @staticmethod
    def print_mem(name, num_bytes):
        print(f"{name}: {num_bytes / 1024 / 1024 / 1024} GB")

    @staticmethod
    def bytes_to_gb(num_bytes):
        # truncate to 3 decimal places
        return round(num_bytes / 1024 / 1024 / 1024, 3)

    @property
    def param_num(self):
        # 4 * h * h + 3 * h * f + 2 * h + 4 * r * h
        return self.L * (4 * self.h * self.h + 3 * self.h * self.f + 2 * self.h + 4 * self.r * self.h)

    @property
    def weights(self):
        # 2 * param_num (only weights, no gradient)
        return self.param_num * 2

    @property
    def gradient(self):
        # 2 * param_num
        return self.param_num * 2

    @property
    def trainable_param_num(self):
        # 2 * 2 * L * r * h, assuming we have only 2 lora (each 2 mats A and B)
        if self.r == 0:
            return self.param_num
        else:
            return 2 * 2 * self.L * self.r * self.h

    @property
    def optimizer_state(self):
        # trainable_param_num * 12
        return self.trainable_param_num * 12

    @property
    def intermediate_activation(self):
        # L*(51*b*s*h + 5*b*a*s*s + 4*b*s*r)
        return self.L * (51 * self.b * self.s * self.h + 5 * self.b * self.a * self.s * self.s + 4 * self.b * self.s * self.r)

    @property
    def total(self):
        return self.weights +self.gradient + self.optimizer_state + self.intermediate_activation

    def report(self):
        self.print_config()
        print("param_num", self.param_num)
        print("trainable_param", self.trainable_param_num)

        tab = tabulate([
            ["weights", self.bytes_to_gb(self.weights)],
            ["gradient", self.bytes_to_gb(self.gradient)],
            ["optimizer state", self.bytes_to_gb(self.optimizer_state)],
            ["intermediate activation", self.bytes_to_gb(self.intermediate_activation)],
            ["total", self.bytes_to_gb(self.total)]
        ], headers=["item", "memory(GB)"], tablefmt="github")

        print(tab)


if __name__ == '__main__':
    m = MemModel(r=0, L=8)
    m.report()
