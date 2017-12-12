## SALTY
[![Build Status](https://travis-ci.org/wesleybeckner/salty.svg?branch=master)](https://travis-ci.org/wesleybeckner/salty)
[![PyPI version](https://badge.fury.io/py/salty-ilthermo.svg)](https://badge.fury.io/py/salty-ilthermo)
========
Salty is an interactive data exploration tool for ionic liquid data from ILThermo (NIST)

# development notes

*Django*
Django is the predominant web framework for Python. Its strengths and weaknesses both draw from its template like approach to web development; it is simple, yet a bit of a curmudgeon when it comes to stylistic or database choices (Django expects SQL). There is no web server built into Django however, making it flexible when it comes time to decide on where to launch the final product.

*Tornado*
The web framework Tornado was developed and released by facebook. It's novelty is it's handling of thousands of simultaneous web connections aka the bundling of its own webserver is inherit in how it works. Tornado's approach to templates are minimal, providing a lot of flexibility. It expects database quieres to be "your problem" and thus offers a lot of structures support for handling this. 

*Flask*
This youngest of the three main web frameworks packs a powerful punch in just 1000 lines of code. Flask provides perhaps the easiest and straightforward approach to view functions for the python-background developer: they are written entirely in python. Flask uses jinja2 as its template engine, one that is also widely adopted by Django developers. Flask is completely agnostic when it comes to databaseand webserver handling/deployment. 
