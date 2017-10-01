### d-vector approach for Speaker Verification implemented in Keras  


Reference for DNN:
Variani, Ehsan, Xin Lei, Erik McDermott, Ignacio Lopez Moreno, and Javier Gonzalez-Dominguez. "Deep neural networks for small footprint text-dependent speaker verification." In Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on, pp. 4052-4056. IEEE, 2014. [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf)

Reference for CNN:
Chen, Y. H., Lopez-Moreno, I., Sainath, T. N., Visontai, M., Alvarez, R., & Parada, C. (2015). Locally-connected and convolutional neural networks for small footprint speaker recognition. In Sixteenth Annual Conference of the International Speech Communication Association. [paper](https://pdfs.semanticscholar.org/ef8d/6c4c65a9a227f63f857fcb789db4202f2180.pdf)


__data:__ WSJ and LibriSpeech Corpus  
__features:__ 32 dimensional log filterbank generated using HTK Toolkit  
__labels:__ labels are force aligned using ASR Model built using Kaldi's WSJ recipe.  


Work was done at [Learning and Extraction of Acoustic Pattern Lab, IISc](http://leap.ee.iisc.ac.in) under the guidance of [Prof. Sriram Ganapathy](http://leap.ee.iisc.ac.in/sriram)
