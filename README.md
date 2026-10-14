# Graphoskop

## Why Graphoskop

Graphoskop is a Python library for network analysis built around three ideas.

**Node attributes are first-class citizens.** Real networks come with information about their
nodes: ages, opinions, cities, categories. In Graphoskop, numeric and qualitative node attributes
are loaded together with the edges and stay attached to the nodes through every operation, so
that they can be analyzed with the network rather than next to it: how similar connected nodes
are, how a variable spreads or varies over the network, how communities relate to attributes.

**GPUs are native.** Graphoskop stores networks as PyTorch and PyTorch Geometric tensors, so the
same code runs on a CPU or on a GPU just by choosing the device when loading a network. Laplacian
operations, random walks, centralities and network generators are written to run on the device.

**It is a pedagogical companion to the textbook.** Graphoskop follows
[The Atlas for the Aspiring Network Scientist](https://www.networkatlas.eu/): its functions use
the concepts and the vocabulary of the book, so that students can try out what they read.

## Installation

Graphoskop requires Python 3.9 or later. Install it with pip:

    pip install graphoskop

This also installs its dependencies:

* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [SciPy](https://scipy.org/)
* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

To use a GPU, install the build of PyTorch matching your hardware first, following the
instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Usage

A network is loaded from an edge list and, optionally, a table of node attributes, as files or as
pandas DataFrames. For example, with a tab-separated edge list `edges.tsv`:

    source   target
    alice    carol
    carol    eve
    eve      alice
    bob      dave
    carol    bob

and a tab-separated node table `nodes.tsv`:

    node     age   city
    alice    34    Rome
    bob      29    Oslo
    carol    41    Rome
    dave     35    Oslo
    eve      52    Rome

one can check whether people tend to be connected to people of the same city:

    import graphoskop as gsk

    net = gsk.load_network("edges.tsv", "nodes.tsv")    # add device = "cuda" to work on a GPU
    gsk.assortativity(net, "city")                      # 0.52: same-city links are more common than by chance

Everything else, from centralities and communities to node attribute analysis and network
generators, is described in the documentation:
[graphoskop.readthedocs.io](https://graphoskop.readthedocs.io/en/latest/).

## Performance

