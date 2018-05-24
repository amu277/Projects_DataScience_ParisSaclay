from betfair_extract_defs import *

# Get directory of files
dir = 'C:/Users/amu277/Documents/futbol/Betfair/'

# Obtain filelist with files greater than 11 kb (more likely to be complete)
filelist = [dir +file for file in os.listdir(dir) if os.path.getsize(dir +file) > 20000]

# Print filelist length
print(len(filelist))


# Setup dataframe to collect information

# Create csv file to append information to and open here:
with open('C:/Users/amu277/Documents/futbol/Betfair/2016_mkt_ltp_betfair.csv', 'a') as f:
    # Open multiple files
    # def json_extract(filelist):

    file_var = 0

    for file in filelist:

        # try:
        #     filedata = open(dir +file, encoding='latin_1').read()

        # except:
        #     filedata = open(dir +file, encoding='utf_8').read()

        # Open multi-line file and get info
        mkinfo = []
        for line in open(file, 'r'):
            mkinfo.append(json.loads(line))

        # File info
        mkl = len(mkinfo)
        filename = os.path.basename(file)
        file_var += 1

        # Get event and market information
        mid, mklist, tm1, tm2, draw = get_event_info(mkinfo, mkl)

        # Get indexes of where market is inPlay = True/False
        list_no, list_nof = get_inplay_indices(mkinfo, mkl)

        if list_no and list_nof and list_no != [0] and list_nof != [0]:
            idx_fp = min(list_no)  # first index where inPlay is True (+1 to include in range)
            idx_lp = max(list_nof)  # last index where inPlay is False

            # Check indexes
            if (idx_lp > idx_fp) or ((idx_lp + 1) == idx_fp) or ((idx_fp + 1) == idx_lp):
                pass

            # Get final trading ltps
            else:
                ltps, rdf, idx_fp, idx_lp = get_final_ltps(list_no, list_nof, mkinfo)

                try:
                    # Append information and get market_df
                    market_df = get_market_df(mid, ltps, mklist, tm1, tm2, draw, idx_lp, idx_fp, mkl, file_var, filename)

                    # Check probabilities add up to 1
                    if (0.98 <= sum(market_df.prob) <= 1.015) and (2 <= len(market_df) <= 3):
                        market_df.to_csv(f, header=False)  # Append to csv
                        print('file number {} processed, name: {}'.format(file_var, filename))

                    else:
                        print('Unusual results for file {}'.format(filename))

                except:
                    pass
