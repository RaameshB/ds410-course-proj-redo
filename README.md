# ds410-course-proj-redo

I have example code up in examples/data_proc_example.ipynb

you don't need to run the get data script, i already have the necessary files downloaded. we need to modify the script if we want to get the data from a different month though

look at the jupyter notebook in the playground folder for my work in extracting the plaintext WET file in parallel

also request at least 16 GB of ram on ROAR when you're trying to run the notebook

## REPO STRUCTURE
ds410-course-proj-redo
|-- data: this is where downloaded data is stored. right now it is just the paths files for the WARC, WAT, and WET files on Common Crawl
    |-- warc.paths: this file is where the paths to the WARC files on the Common Crawl servers reside
    |-- wat.paths: same as warc.paths but for wat files
    |-- wet.paths: same as warc.paths and wac.paths but for wet files
|-- dataproc: this is where the data processing utils I've made are stored
    |-- dataproctools.py: this is where the utility functions for downloading and processing the WET files reside. also includes helper functions to save and load processed WET file RDDs 
|-- examples: example notebooks to understand the code are stored here
    |-- data_proc_example.ipynb: this is an example of using dataproc functions to download the WET files and process them into a usable string RDD. more will be added to this
|-- sparktools: this is where anything that handles pyspark is stored. it's mostly just lightweight wrappers that make initialization easier
    |-- sparkhandler.py: this just contains a wrapper class spark sessions. gets you a spark context and a spark session, and handles initializing allocated memory and cpu cores in a simple manner
|-- get_data.sh: this downloads the path files from Common Crawl servers. this script will need to be modified in order to do analysis of multiple months
|-- README.md: this file
