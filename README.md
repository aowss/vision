# Vision

## Pre-requisites

* [Python 3.6](https://www.python.org/downloads/) since PyBuilder doesn't support version 3.7
* [PyBuilder](https://github.com/pybuilder/pybuilder)

## Build and Test

* Create a virtual environment

This step is required if you don't want to install the dependencies globally.

> `python3 -m venv env`  
> `source env/bin/activate`

* Install [PyBuilder](http://pybuilder.github.io/)

The project is built using [PyBuilder](http://pybuilder.github.io/).

> `pip install pybuilder`

* Download dependencies

The dependencies are therefore declared in [the `build.py` file](./build.py).  
They can be _installed_ using the following command :
> `pyb install_dependencies`

* Run unit tests

The unit tests are therefore located in [the `unittest` folder](./src/unittest).  
They can be run using the following command :
> `pyb verify`
