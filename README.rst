.. image:: https://img.shields.io/pypi/v/pupil-labs-dynamic-rim.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/pupil-labs-dynamic-rim.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/skeleton

.. image:: https://github.com/pupil-labs/dynamic-rim-module/workflows/tests/badge.svg
   :target: https://github.com/pupil-labs/dynamic-rim-module/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. .. image:: https://readthedocs.org/projects/skeleton/badge/?version=latest
..    :target: https://skeleton.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2022-informational
   :target: https://blog.jaraco.com/skeleton

This package allows you to use the Dynamic RIM module in Pupil Labs.

This is an extension of our `RIM <https://docs.pupil-labs.com/invisible/explainers/enrichments/#reference-image-mapper>`__
enrichment that allows you to select a screen/display in the reference image and plot the gaze over the content displayed on it.
It will also give you a csv file with the coordinates of the gaze in the screen coordinates.

To install it, use:

..  code-block:: python

    pip install pupil-labs-dynamic-rim

To run it:

..  code-block:: python

    pl-dynamic-rim

See our `docs <https://docs.pupil-labs.com/>`__ for more information.
