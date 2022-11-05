#!/bin/bash

rm -f ../log/*

echo 'copy prod cfg'
cp ../cfg/ioq.cfg.prod ../cfg/ioq.cfg

echo 'start ioq_server'
nohup /home/keplercapital2/anaconda2/bin/python ioq_server.py >>../log/ioq.out 2>>../log/ioq.err &


