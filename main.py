# -*- coding: utf-8 -*-

"""
Created on January 27, 2021
@author: Charlie Bailly <charlie.bailly@telecom-paris.fr>
@author: Alexis Barreaux <alexis.barreaux@telecom-paris.fr>
"""

# scapy documentation : https://scapy.readthedocs.io/en/latest/
# sous windows il faut installer npcap
#  a, b = sr(requête) mets les requêtes qui ont eu une réponse dans a et les autres dans b
# (1,20) = de 1 à 20

import scapy
from scapy.layers.http import HTTP, HTTPRequest
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.layers.tls.automaton_cli import TLSClientAutomaton
from scapy.layers.tls.handshake import TLSClientHello
# TODO il faut trouver comment ce servir de ça
from scapy.layers.tls.extensions import TLS_Ext_Heartbeat
from scapy.main import load_layer
from scapy.sendrecv import sniff, sendp

from scapy.sessions import TCPSession
# TODO apparemment TCPSession permet de gérer un peu le TLS, TLSSession sinon

# Ajout de TLS
load_layer('tls')


def test_pkt(dst="127.0.0.1"):
    trame = Ether()
    ip = IP(src="8.8.8.8", dst=dst)
    tcp = TCP(dport=443, flag="S")

    test = trame / ip / tcp
    sendp(test)
    return


def https_test():
    # dans a on peut utiliser version"sslv2" ou "tls1" ou "tls12" pour faire des tests.
    # data sert à envoyer des données après que le lien soit fait
    a = TLSClientAutomaton.tlslink(HTTP, server="www.google.com", dport=443)
    pkt = a.sr1(HTTP() / HTTPRequest(), session=TCPSession(app=True), timeout=2)
    pkt.show()

    return
