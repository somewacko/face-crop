# Face Crop!

Crops people's faces out of a video when they're most likely speaking. Looks
for sequences that meet two conditions:

* There is exactly one face detected throughout the sequence is roughly the
  same spot.
* Most of the corresponding audio frames have a max amplitude over some
  threshold.

...and then crops them out and writes the result to a video file.

Of course, this assumes that the videos consist of mostly people speaking.
(news, speeches, vlogs, etc.) Don't expect this to work on videos that don't
meet this assumption. (or do it anyway!)

*I made this as a quick script to do some video processing for a research
project. This is not quality code!*

Python 2 because I have to. Requires ffmpeg, OpenCV, and SciPy.

