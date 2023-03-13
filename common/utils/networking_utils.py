# -----------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
# -----------------------------------------------------------------------------
import socket

# NOTE  will be configured via xml files
steering_subscription_topic = b'steering'

# NOTE  will be configured via xml files
default_range_of_ports = {
                          # range of ports for Orchestrator
                          'ORCHESTRATOR': {'MIN': 59100,
                                           'MAX': 59120,
                                           'MAX_TRIES': 20},
                          # range of ports for Command&Control
                          'COMMAND_CONTROL': {'MIN': 59121,
                                              'MAX': 59150,
                                              'MAX_TRIES': 30},
                          # range of ports for Application Companions
                          'APPLICATION_COMPANION': {'MIN': 59150,
                                                    'MAX': 59200,
                                                    'MAX_TRIES': 50},
                          # range of ports for Application Managers
                          'APPLICATION_MANAGER': {'MIN': 59201,
                                                  'MAX': 59300,
                                                  'MAX_TRIES': 100}
                          }


def my_ip():
    """returns the ip address where the calling process is running"""
    my_host_name = socket.gethostname()
    my_ip = socket.gethostbyname(my_host_name)
    return my_ip


def my_host_name():
    """returns the hostname of the calling process is running"""
    return socket.gethostname()