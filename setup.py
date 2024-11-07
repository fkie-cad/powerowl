from setuptools import setup, find_namespace_packages

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
    version="0.1.0",
    packages=find_namespace_packages(include=[
        'powerowl.*'
    ]),
    install_requires=[
        'ipaddress==1.0.23',
        'matplotlib>=3.1.2',
        'netifaces==0.11.0',
        'networkx>=2.5',
        'numpy',
        'pandapower==2.14.1',
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
