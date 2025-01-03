from setuptools import setup, find_namespace_packages, find_packages

"""

########     #######    ##      ##   ########   ########      ^___^     ##      ##   ##       
##     ##   ##     ##   ##  ##  ##   ##         ##     ##    ##   ##    ##  ##  ##   ##       
##     ##   ##     ##   ##  ##  ##   ##         ##     ##   ## O O ##   ##  ##  ##   ##       
########    ##     ##   ##  ##  ##   ######     ########    ##  V  ##   ##  ##  ##   ##       
##          ##     ##   ##  ##  ##   ##         ##   ##     ##     ##   ##  ##  ##   ##       
##          ##     ##   ##  ##  ##   ##         ##    ##     ##   ##    ##  ##  ##   ##       
##           #######     ###  ###    ########   ##     ##     #####      ###  ###    ######## 

"""


setup(
    name='powerowl',
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        'ipaddress>=1.0.23',
        'matplotlib>=3.1.2',
        'netifaces>=0.11.0',
        'networkx>=2.5',
        'numpy',
        'pandapower==2.14.6',
        'pandas',
        'pydot',
        'plotly',
        'igraph==0.9.9',
        'pyyaml',
        'setuptools',
        'shapely',
        'ifcfg'
    ],
    python_requires=">=3.10.0",
    author="Lennart Bader (Fraunhofer FKIE)",
    author_email="lennart.bader@fkie.fraunhofer.de",
)
