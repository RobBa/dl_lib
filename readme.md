

# Troubleshooting


### Building on Windows

The implementation of the Python wrapper does not work on MSVC6/7 in its current form. This is due to an issue that arises from Boost Python in combination with these compilers. Workarounds are proposed, but not implemented. More information here [here](https://beta.boost.org/doc/libs/develop/libs/python/doc/html/tutorial/tutorial/exposing.html).