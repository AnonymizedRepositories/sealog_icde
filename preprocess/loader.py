import os
import re
import pandas as pd
import multiprocessing


import signal

def handler(signum, frame):
    raise TimeoutError("Timeout occurred")

# Set the timeout duration in seconds
timeout_duration = 1
# Set the signal handler
signal.signal(signal.SIGALRM, handler)


def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex
    

def process_chunk(chunk, regex, headers):
    """ Function to process a log file chunk and return a dataframe 
    """
    log_messages = []
    linecount = 0
    for line in chunk:
        try:
            signal.alarm(timeout_duration)
            match = regex.search(line.strip())
            message = [match.group(header) for header in headers]
            log_messages.append(message)
            linecount += 1
            signal.alarm(0)
        except TimeoutError as e:
            print("Timeout occurred for line {}.".format(line))
        except Exception as e:
            pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    # print("{} loading chunk done.".format(os.getpid()))
    return logdf

def log_to_dataframe(log_file, regex, headers, chunksize=500000, num_processes=None):
    """ Function to transform log file to dataframe using multi-processing
    """
    # open the log file
    with open(log_file, 'r', newline="\n") as fin:
        # read the log file in chunks
        chunks = iter(lambda: list(fin.readlines(chunksize)), [])
        # process each chunk in a separate process
        pool = multiprocessing.Pool(num_processes)
        results = []
        for chunk in chunks:
            result = pool.apply_async(process_chunk, args=(chunk, regex, headers))
            results.append(result)
        pool.close()
        pool.join()
        # combine the results from each process into a single dataframe
        logdf = pd.concat([result.get() for result in results], ignore_index=True)

    logdf["LineId"] = [i+1 for i in range(logdf.shape[0])]
    print("{} lines of formated log read.".format(len(logdf)))
    return logdf


def load_raw_log(log_format, log_file, chunksize=500000, num_processes=None):
    print(f"Loading data with {num_processes} processes, chunksize={chunksize}")
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(log_file, regex, headers, chunksize=chunksize, num_processes=num_processes)
    return df_log