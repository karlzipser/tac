# python3 utilz2/dev/project.py --src tac_ideal --tag something
from utilz2 import *
if __name__ == '__main__':
    import sys
    print("Argument List:", str(sys.argv))
    s=select_from_list(['classifier','test'])
    if s=='classifier':
        from .classifier import *
    if s=='test':
        from .test import *
    else:
        assert False
#EOF