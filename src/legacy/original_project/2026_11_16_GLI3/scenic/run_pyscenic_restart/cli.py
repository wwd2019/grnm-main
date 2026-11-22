import numpy as np
np.object = object
np.bool = bool
np.int = int
np.float = float
if __name__ == '__main__':
    from pyscenic.cli.pyscenic import main
    main()
