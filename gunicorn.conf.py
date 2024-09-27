timeout = 300
workers = 8
bind = "0.0.0.0:5491"
worker_class = "gevent"
pythonpath = "."
app_module = "main:app"
reload = True  # Set to False in production
   