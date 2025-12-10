import math
import shutil
import os

DOMAINSPERWET = 20000
def get_extracted_wet(spark_context, approx_sample_size, wet_paths_file = "data/wet.paths", seed = 0):
    sc = spark_context
    wets_to_load = math.ceil(approx_sample_size / DOMAINSPERWET)
    paths_rdd = sc.textFile(wet_paths_file)
    sampled_paths = sc.parallelize(paths_rdd.takeSample(withReplacement=False, num=wets_to_load, seed=seed))
    url_head = "https://data.commoncrawl.org/"
    urls_rdd = sampled_paths.map(lambda x: url_head + x)

    # we use requests to get a http response from the common crawl server
    def get_request(url):
        import requests
        import random
        import time

        jitter = random.random()
        time.sleep(jitter)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return response
        else:
            # Here we implement an exponential back off strategy to make sure we're not overloading the server
            # Here we wait for a total of (2^11)/10 = 204.8 seconds or about 3.5 minutes max to get a response from the server
            # So total wait time can be a bit over 5 minutes
            for i in range(12):
                jitter = random.random()
                time.sleep((2 ** i + jitter) / 10)
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    # Request was successful
                    return response
            return None

            # This is to have our responses RDD be filtered so we don't have any Nonetypes

    responses_rdd = urls_rdd.map(get_request).filter(lambda x: x is not None)

    # we convert the responses into gzip files in memory
    def get_zipped(response):
        import io
        return io.BytesIO(response.content)

    zipped_files = responses_rdd.map(get_zipped)

    # we unzip the zipped files also within memory
    def unzip(zipped):
        import gzip
        with gzip.GzipFile(fileobj=zipped) as decompressed_file:
            return decompressed_file.read()

    unzipped_rdd = zipped_files.map(unzip).persist()

    str_rdd = unzipped_rdd.map(lambda x: x.decode("utf-8"))

    return str_rdd

def save_rdd(rdd, path_to_save, overwrite=False):
    if overwrite and os.path.exists(path_to_save):
        shutil.rmtree(path_to_save)
    rdd.saveAsPickleFile(path_to_save)

def load_rdd(spark_context, path_to_load):
    return spark_context.pickleFile(path_to_load)

def extracted_wet_to_df(spark_session, extracted_wet_rdd):
    ss = spark_session
    def split_sites(string):
        import re
        # The header of every item is WARC/1.0\r\nWARC-Type: conversion, but we assume that the WARC version could be different
        return re.split(r"WARC/\d.\d\r\nWARC-Type: conversion\r\n", string)[1:]

    split_rdd = extracted_wet_rdd.flatMap(split_sites)

    def regex_split_to_kv(string):
        import re
        return re.split(r"Content-Length: \d+\r\n", string)
    KV_mapped = split_rdd.map(lambda x: tuple(regex_split_to_kv(x)))

    Ksplit = KV_mapped.map(lambda x: (x[0].split('\r\n'), x[1]))

    def get_warc_target_uri(split_key):
        return split_key[0].split(': ')[1]

    def get_warc_target_date(split_key):
        return split_key[1].split(': ')[1]

    def get_warc_record_id(split_key):
        return split_key[2].split(': ')[1]

    def get_warc_languages(split_key):
        return split_key[5].split(': ')[1].split(',')

    def get_tld_from_url(url):
        import re
        # search for strings that start with "." and are followed by a "/" that are at least 2 letters long and also search for also that russian tld approximation that makes no sense
        searched = re.search(string=url, pattern=r"\.((?:xn--[a-z0-9]+)|[a-z]{2,})(?=[/?#:]|$)")
        return searched[0] if searched is not None else "None"

    with_processed_keys = Ksplit.map(lambda x:
                                     (
                                         get_warc_record_id(x[0]),
                                         get_warc_target_uri(x[0]),
                                         get_warc_target_date(x[0]),
                                         get_warc_languages(x[0]),
                                         x[1]
                                     )
                                     )

    tlds_added = with_processed_keys.map(lambda x: (x[0], x[1], x[2], x[3], get_tld_from_url(x[1]), x[4]))
    return ss.createDataFrame(tlds_added, schema=['warc_id', 'target_uri', 'date', 'languages', 'tld', 'raw_content'])