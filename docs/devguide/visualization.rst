Visualizations
--------------

Video Conversion and Docs
=========================

Videos displayed in the docs were recorded with Apple's screen grabbing tool
and converted to mp4 using ``ffmpeg`` with quality level ``-q:v 15`` 
(levels vary between 0 (best) and 31 (worst). They need to be copied to the 
``docs/_static`` directory and referenced in the docs with relative paths

as

.. code:: rst

    .. video:: ../_static/file.mp4  

Here's three commands I usually use:

.. code:: bash

    mv Screen\ Recording\ 2020-08-03\ at\ 16.56.15.mov lattice_SIR.mov
    ffmpeg -i lattice_SIR.mov -q:v 15 lattice_SIR.mp4
    mv lattice_SIR.mp4 ~/forschung/disease_dynamics/epipack/docs/_static/
