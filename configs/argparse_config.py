import argparse

parser = argparse.ArgumentParser(description='arg_reader')
parser.add_argument('-m', dest='mode', default='masks', type=str)
parser.add_argument('-s', dest='statistic', default='disable', type=str)

reminder = '\n\n*\n You can use: \
            \n-m masks or -m dist for mode, \
            \n-s enable or -s disable for statistics\n*\n'