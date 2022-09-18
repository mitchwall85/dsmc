from pint import UnitRegistry
import numpy as np

ureg = UnitRegistry()



TUBE_D = 2*ureg.mm
TUBE_L = np.array([1, 5, 10, 20])*ureg.mm


if __name__ == "__main__":
     a = 1
     print(TUBE_L)

     dl = 0.1*ureg.mm
     dd = 0.2*ureg.mm

     # inlet domain



     for tube_len in TUBE_L:
          a = 1