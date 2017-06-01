# salty
An interactive data exploration tool for json structured ionic liquid data from ILThermo (NIST)

# timeline
4/13-got rdkit working on mac, created github repo

4/20-load the json files into pandas dataframe

4/26-we'll have 3 trained models: ANN, LASSO lars IC, SVM

5/10-do some javascript/css/web design tutorials setup AWS webserver for our project.
	We'll have done some additional model training, data visualization, data exploration

5/24-have at least a few visualizations in js and a dashboard outline for the web app

6/7-some functional web app is somewhat ready. Maybe 3 models available, a clever way to 
	select data and interact with it. Maybe some advanced statistical analysis tools.

# deliverables

# development notes

*Django*
Django is the predominant web framework for Python. Its strengths and weaknesses both draw from its template like approach to web development; it is simple, yet a bit of a curmudgeon when it comes to stylistic or database choices (Django expects SQL). There is no web server built into Django however, making it flexible when it comes time to decide on where to launch the final product.

*Tornado*
The web framework Tornado was developed and released by facebook. It's novelty is it's handling of thousands of simultaneous web connections aka the bundling of its own webserver is inherit in how it works. Tornado's approach to templates are minimal, providing a lot of flexibility. It expects database quieres to be "your problem" and thus offers a lot of structures support for handling this. 

*Flask*
This youngest of the three main web frameworks packs a powerful punch in just 1000 lines of code. Flask provides perhaps the easiest and straightforward approach to view functions for the python-background developer: they are written entirely in python. Flask uses jinja2 as its template engine, one that is also widely adopted by Django developers. Flask is completely agnostic when it comes to databaseand webserver handling/deployment. 
