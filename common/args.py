# -----------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# -----------------------------------------------------------------------------
import argparse
from pathlib import Path

from common.utils.common_utils import strtobool


def xml_file_exists(path_and_filename):
    """
    Returns the path to given application if it exists at given location.
    Raises an exception if the application does not exists at specified
    location.

    Parameters
    ----------
        path_and_appname: str
            Location and application name

    Returns
    ------
        path_to_app: Path
            Path to the application if it exists.
            Otherwise, it raises ArgumentTypeError exception.
    """
    path_to_app = Path(path_and_filename)
    if path_to_app.is_file():
        return path_to_app
    else:
        raise argparse.ArgumentTypeError(f'Does not exist: <{path_to_app}>')

def get_parser():
    '''
    creates and returns an object of ArgumentParser to parse the command line
    into Python data types.
    '''
    return argparse.ArgumentParser(
                    prog='MSM',
                    usage='%(prog)s --interactive (optional) --action-plan <path/to/plan.xml> --global-settings <path/to/settings.xml>',
                    description='Launch a co-simulation workflow defined in XML file specified in --action-plan. '
                                'steering is interactive if --interactive (optional) is set.',
                    formatter_class=argparse.RawTextHelpFormatter)

def add_CLI_arguments(parser):
    '''
    Fills ArgumentParser to take CLI arguments for parsing.

    Parameters
    ----------
        parser: ArgumentParser
            ArgumentParser object to fill with the program arguments
    '''
    # i. if interactive steering is enabled
    parser.add_argument(
        '--interactive',
        '-i',
        help='(optional) Activate interactive steering. Default is false.',
        metavar='is_interactive',
        type=strtobool,
        nargs='?',
        const=True,
        default=False,
        required=False,
    )
    
    # ii. path to co-simulaiton plan XML file
    parser.add_argument(
        '--action-plan',
        '-a',
        help='XML file defining the Co-Simulations Plan to be executed',
        metavar='co_simulation_plan.xml',
        type=xml_file_exists,
        required=True,
    )

    # iii. path to global-settings XML file
    parser.add_argument(
        '--global-settings',
        '-g',
        help='XML file defining the common settings for Co-Simulation',
        metavar='co_simulation_global_settings.xml',
        type=xml_file_exists,
        required=True,
    )


def get_parsed_CLI_arguments():
    """
    Parses the command-line arguments passed to the Modular Science Manager.

    Returns
    ------
        parsed_arguments: argparse.Namespac
            parsed arguments into Python data types
    """
    # create a parser
    parser = get_parser()
    # fill ArgumentParser to take CLI arguments for parsing
    add_CLI_arguments(parser)
    # return parsed CLI arguments
    return parser.parse_args()