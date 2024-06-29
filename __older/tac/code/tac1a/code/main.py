# python3 utilz2/dev/project.py --src tac_ideal --tag something
from utilz2 import *
if __name__ == '__main__':
    import sys
    print("Argument List:", str(sys.argv))
    s=select_from_list(['classifier'])
    if s=='classifier':
        from .classifier import *
    else:
        assert False
#EOF