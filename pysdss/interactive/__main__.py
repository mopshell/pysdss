import argparse

from stars.interface import TextInterface


def _launch():

    #parser = argparse.ArgumentParser(prog = 'py -3.5 -m stars',description = "start the Stars command line")
    #parser.add_argument('-k', '--keep-going',
    #                    action = 'store_true',
    #                    default = False)

    #parser.add_argument('filename')   
    #args = parser.parse_args()
    #iface = TextInterface(args.keep_going, args.filename)

    iface = TextInterface()
    iface.cmdloop()


if __name__ == '__main__':
    _launch()
