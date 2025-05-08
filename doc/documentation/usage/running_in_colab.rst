======================================
Running whobpyt in google colab
======================================

Google colab is an increasingly convenient and useful scientific omputing environment. 

WhoBPyT can be used in google colab with some minimal additional steps. 

First, in a browser tab, navigate to https://colab.research.google.com/github/GriffithsLab/whobpyt , 
selecting organization GriffithsLab, repository whobpyt, and branch gh-pages. 

You should then see a drop-down list of .ipynb files, corresponding to each of the examples in the sphinx-gallery docs. 

Select one of these, and approve the 'run anyway' option. 

Then, insert and run the following in a cell at the top:


.. code::

    import os,time
    os.system('rm -rf whobpyt')
    os.system('git clone https://github.com/griffithslab/whobpyt')
    time.sleep(3)
    os.chdir('whobpyt')
    time.sleep(3)
    os.system('python install_colab.py')    
    
Now you should be good to continue with the rest of the example code in the notebook, and experiment with new ideas. 

