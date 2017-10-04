#from __future__ import absolute_import, unicode_literals
from geoprocessing import app
import time


@app.task
def add(x, y):
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def xsum(numbers):
    return sum(numbers)


@app.task(bind=True)
def info(self,a,b):
    print('Executing task id {0.id}, args: {0.args!r} kwargs: {0.kwargs!r}'.
          format(self.request))


@app.task
def sleep():
    time.sleep(10)
