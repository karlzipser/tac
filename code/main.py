# python3 utilz2/dev/project.py --src tac_ideal --tag something
from utilz2 import *
if __name__ == '__main__':
    import sys
    print("Argument List:", str(sys.argv))
    s=select_from_list(['classifier','gen_classifier'])
    if s=='findideal':
        from .classifier import *
    elif s=='gen_classifier':
        from .gen_classifier import *
    else:
        assert False
#EOF