
===
MMA
===

MMA is a command line sample conversion tool created
to transform .SFZ sample packs to .XI (Fasttracker 2 eXtended Instrument)
format, supported by a number of music creation software.

Designed to deal with MilkyTracker_ music tracker.

Based on Samplicity_ version 0.4 by Andrii Magalich.

Thanks to `Alex Zolotov`_ for help and materials.

Why MMA?
========

I forked Samplicity_ instead of contributing simply because we have different
goals: while Samplicity_ is designed in order to be compatible with `SunVox tracker`_,
with its advanced features like stereo or 32-bit samples, my goal is to be compatible
with MilkyTracker_, which has many more limitations, like no more than 16 samples per instrument.

I chose to fork version 0.4 instead of version 0.5 because version 0.5 depends on the
scikits.audiolab library, which is ridiculously hard to install; so another goal of MMA
that is different from Samplicity_ is to depend only on the Python_ standard library.

Disclaimer
==========

MMA does not yet support all the features in intersection of .SFZ and .XI.

Now it is tested **only** in MilkyTracker_ with the |SSO|_ sample pack.

    Your tracker may crash for wrongly encoded .XI-instruments, so
    **you should save your files every time before loading an instrument**

Before running MMA it is recommended to use a tool like SOX_ to convert all the samples to 8 bit or 16 bit mono wav format.

Known Issues
------------

This version of MMA has the following known issues:

- Volume envelope times may be wrong
- Auto vibrato is not supported yet
- Sample looping is not supported yet

Status and Roadmap
------------------

MMA version 0.1.0 supports all the .XI compatible opcodes of the |SSO|_ sample pack,
but in order to support auto vibrato and sample looping, it has to be tested with other
sample packs, so in the following days I will test the |Aria|_ sample pack in order to
implement the missing functionality.

Formats
=======

eXtended Instrument
-------------------

This format was created in 1990's for DOS music tracker called Fasttracker 2.
It's binary, old and rusty, but still useful.

SFZ
---

Open format by Cakewalk company. Designed for creation in notepad. Sample pack
contains .sfz textfile and a number of samples nearby. So, you can create
your sample pack without any specific software. See more `here
<http://www.cakewalk.com/DevXchange/article.aspx?aid=108>`_

Usage
=====

MMA is writted in Python_ v2.7. To use this tool Python_ v2.7+ should be
installed on your computer.

Sample conversion
-----------------

To convert single sample pack, navigate in **terminal/bash/command** line to
sample pack folder and run the following command:

.. code-block:: bash

    python "<PATH TO MMA FOLDER>/mma.py" "<SAMPLE PACK NAME>.sfz"

If Python is installed, path to MMA is right and sample pack is a valid
.SFZ file, you'll see something like this:

.. code-block:: bash

    --------------------------------------------------------------------------------
    Converting " Keys - Grand Piano (Forte).sfz "
    --------------------------------------------------------------------------------
    * 16 bit stereo sample " samples/grand piano/piano-f-c1.wav " 726 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-d#1.wav " 735 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-f#1.wav " 734 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-a1.wav " 732 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-c2.wav " 725 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-d#2.wav " 709 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-f#2.wav " 700 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-a2.wav " 695 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-c3.wav " 623 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-d#3.wav " 639 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-f#3.wav " 607 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-a3.wav " 577 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-c4.wav " 563 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-d#4.wav " 526 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-f#4.wav " 499 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-a4.wav " 461 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-c5.wav " 441 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-d#5.wav " 410 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-f#5.wav " 361 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-a5.wav " 334 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-c6.wav " 322 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-d#6.wav " 273 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-f#6.wav " 218 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-a6.wav " 204 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-c7.wav " 138 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-d#7.wav " 104 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-f#7.wav " 97 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-a7.wav " 104 kB
    * 16 bit stereo sample " samples/grand piano/piano-f-c8.wav " 103 kB
    ////////////////////////////////////////////////////////////////////////////////
    Notice: some notes are out of range and ignored
    ['c8']
    ////////////////////////////////////////////////////////////////////////////////
    29 samples
    26751 kB written in file " Keys - Grand Piano (Forte).sfz " during 9.435801 seconds

    1 files converted in 9.437803 seconds

Batch conversion
----------------

To convert more than one .SFZ file you can specify as many arguments to MMA as
you want. Or even use a wildcard

.. code-block:: bash

    python "<PATH TO MMA FOLDER>/mma.py" "<SAMPLE 1>.sfz" "<SAMPLE 2>.sfz" "<SAMPLE 3>.sfz"
    python "<PATH TO MMA FOLDER>/mma.py" *.sfz

Reconversion
------------

If there is corresponding to your sample pack .XI file, MMA won't convert it
again. To force reconversion, add ``--force`` attribute:

.. code-block:: bash

    python "<PATH TO MMA FOLDER>/mma.py" --force "<SAMPLE NAME>.sfz"

Package
=======

Repository contains:

- ``mma.py``
- ``xi_reader.py`` — tool to verify your .XI if something went wrong. Usage: ``python "<PATH TO MMA FOLDER>/xi_reader.py" "<SAMPLE NAME>.xi"``. It will show you full info, contained in .XI file (but not samples binary data). It is useful for bugtrack.
- ``xi_specs.txt`` — specifications of eXtended Instrument edited and improved a bit. Thanks `Alex Zolotov`_
- ``Cakewalk DevXchange - Specifications - sfz File Format.pdf`` — specifications of .SFZ saved from Cakewalk `website <http://www.cakewalk.com/DevXchange/article.aspx?aid=108>`_.

Notices and errors
==================

- **Notice: some notes are out of range and ignored** — .XI supports only 96 notes from C0 to B7, so some notes in your sample pack cannot fit in this range. Consider editing .SFZ file.
- **Notice: some regions are overlapping and would be overwritten** — .SFZ format supports velocity maps. But .XI doesn't. Consider splitting your .SFZ file into separate files. For example, I've got ``Grand Piano (Piano).sfz`` and ``Grand Piano (Forte).sfz``
- **24bit samples are not supported** — .XI and Sunvox don't support 24bit sample format and there is no cooldown feature for them in MMA
- **Too long envelope, shrinked to 512** — .XI does not support envelopes longer than 512 ticks (~10.24 seconds), so you instrument envelope was modified to fit this range
- **Too many samples in file** — .XI does not support more than 16 samples in instrument. Consider splitting your file or removing some.

.. _MilkyTracker: http://milkytracker.org/
.. _Samplicity: https://github.com/ckald/Samplicity/
.. _Alex Zolotov: http://www.warmplace.ru/
.. _Python: https://www.python.org/
.. _SOX: http://sox.sourceforge.net/
.. _`SunVox tracker`: http://www.warmplace.ru/soft/sunvox/

.. |SSO| replace:: Sonatina Symphonic Orchestra
.. _SSO: http://sso.mattiaswestlund.net/

.. |Aria| replace:: Free Sounds for ARIA Engine
.. _Aria: http://www.plogue.com/phpBB3/viewtopic.php?t=7090

