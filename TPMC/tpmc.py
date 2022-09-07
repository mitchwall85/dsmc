from pint import UnitRegistry
import numpy as np

ureg = UnitRegistry()



TUBE_D = 2*ureg.mm
TUBE_L = np.array([1, 5, 10, 20])*ureg.mm


if __name__ == "__main__":
     a = 1
     print(TUBE_L)
    