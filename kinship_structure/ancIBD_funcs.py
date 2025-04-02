"""
Functions for processing and rearranging IBD results from ancIBD 
(Ringbauer et al., Accurate detection of identity-by-descent segments in human ancient DNA, Nat Genet 2023)
(https://github.com/hringbauer/ancIBD).

Mainly adapted from https://github.com/hringbauer/ancIBD/blob/master/notebook/pedsim/process_pedsim.ipynb, with minor changes.
"""

import pandas as pd
import numpy as np
import h5py as h5py
from scipy import interpolate


def filter_ibd_df(df, min_cm=4, snp_cm=60, output=True):
    """Post Process ROH Dataframe. Filter to rows that are okay.
    min_cm: Minimum Length in CentiMorgan
    snp_cm: How many SNPs per CentiMorgan"""

    # Filter for Length:
    length_okay = (df["lengthM"] * 100) > min_cm
    
    # Filter for SNP Density:
    densities = df["length"] / (df["lengthM"] * 100)
    densities_ok = (densities > snp_cm)
    df["SNP_Dens"] = densities
    
    df = df[densities_ok & length_okay].copy()
    
    if output==True:
        print(f"> {min_cm} cM: {np.sum(length_okay)}/{len(length_okay)}")
        print(f"Of these with suff. SNPs per cM> {snp_cm}: \
              {np.sum(densities_ok & length_okay)}/{np.sum(length_okay)}")   
    return df

def roh_statistic_df(df, min_cm=0, col_lengthM="lengthM"):
    """Gives out Summary statistic of ROH df"""
    if len(df)==0:   # If no ROH Block found
        max_roh, sum_roh, n_roh = 0, 0, 0
    
    else:
        idx = df["lengthM"]>min_cm/100.0 # convert to Morgan
        if np.sum(idx)==0:
            max_roh, sum_roh, n_roh = 0, 0, 0
            
        else:
            l = df["lengthM"][idx]
            max_roh = np.max(l)
            sum_roh = np.sum(l)
            n_roh = len(l) 
    return sum_roh, n_roh, max_roh

def roh_statistics_df(df, min_cms=[8, 12, 16, 20], col_lengthM="lengthM"):
    """Gives out IBD df row summary statistics.
    Return list of sum_roh, n_roh, max_roh for each of them [as list]
    min_cm: List of minimum IBD lengths [in cM]"""  
    res = [roh_statistic_df(df, c, col_lengthM) for c in min_cms]
    return res

def load_segment_file(path_segments="../ped-sim/output/output.seg",
                      cm_fac=0.01):
    """Load and return segment File of IBD & ROH blocks.
    Return Pandas dataframe. 
    cm_fac: Factor with which to multiply genetic length columns"""
    df = pd.read_csv(path_segments, sep="\t", header=None)
    df.columns = ["iid1", "iid2", "ch", "Start", "End", 
                  "ibd_stat", "StartM", "EndM", "lengthM"]
    df["length"] = (df["End"] - df["Start"])
    
    for col in ["StartM", "EndM", "lengthM"]:
        df[col] = df[col] * cm_fac  # Correct that original is in cm
    return df

def merge_called_blocks(df, output=False):
        """Merge Blocks in Dataframe df and return merged Dataframe.
        Gap is given in Morgan"""
        if len(df) == 0:
            return df  # In case of empty dataframe don't do anything

        df_n = df.drop(df.index)  # Create New Data frame with all raws removed
        row_c = df.iloc[0, :].copy()
        #row_c["lengthM"] = row_c["EndM"] - row_c["StartM"] # Should be there

        # Iterate over all further rows, update blocks if gaps small enough
        for index, row in df.iloc[1:,:].iterrows():
            ### Calculate Conditions
            con1 = (row["Start"] == row_c["End"]+1)
            con2 = row["ch"] == row_c["ch"]
            con3 = row["iid1"] == row_c["iid1"]
            con4 = row["iid2"] == row_c["iid2"]
            
            if con1 & con2 & con3 & con4:
                row_c["End"] = row["End"]
                row_c["EndM"] = row["EndM"]
                row_c["length"] = row_c["End"] - row_c["Start"]
                row_c["lengthM"] = row_c["EndM"] - row_c["StartM"]

            else:  # Save and go to next row
                df_n.loc[len(df_n)] = row_c  # Append a row to new df
                row_c = row.copy()

        df_n.loc[len(df_n)] = row_c   # Append the last row

        if output == True:
            print(f"Merged n={len(df) - len(df_n)} gaps")
        return df_n
    
def cap_ibd_boarders(df, chs = range(1,23), 
                     h5_path = "/n/groups/reich/hringbauer/git/hapBLOCK/data/hdf5/1240k_v54.1/ch"):
    """Cuts IBD segment file for ch in chs to boundaries matching h5 in h5_path"""
    
    for ch in chs:
        with h5py.File(f"{h5_path}{ch}.h5", "r") as f: # Load for Sanity Check. See below!
            min_map, max_map =  f["variants/MAP"][0],f["variants/MAP"][-1]

        idx_ch = df["ch"]==ch ## Find all segments on chromosome

        ### Cut to Start Positions
        idx = df["StartM"]<min_map
        df.loc[idx_ch & idx, "StartM"] = min_map
        idx = df["EndM"]<min_map
        df.loc[idx_ch & idx, "EndM"] = min_map

        ### Cut to End Positions
        idx = df["StartM"]>max_map
        df.loc[idx_ch & idx, "StartM"] = max_map
        idx = df["EndM"]>max_map
        df.loc[idx_ch & idx, "EndM"] = max_map

    df["LengthM"]= df["EndM"]-df["StartM"] # Update IBD Length
    ### Remove IBD segments fully out of boarder
    idx = df["LengthM"]==0
    df = df[~idx].copy().reset_index(drop=True) 
    return df

def transform_to_snp_pos(df, chs=range(1,23), 
                         h5_path = "/n/groups/reich/hringbauer/git/hapBLOCK/data/hdf5/1240k_v54.1/ch"):
    """Transform positions in IBD dataframe to positions matching indices in 1240k file"""
    
    for ch in chs:
        with h5py.File(f"{h5_path}{ch}.h5", "r") as f: # Load for Sanity Check. See below!
                m = f["variants/MAP"][:]
        p = np.arange(len(m))
        f = interpolate.interp1d(m, p)
        
        ### Map to approximate index positions
        idx_ch = df["ch"]==ch ## Find all segments on chromosome
        df.loc[idx_ch, "Start"] = f(df["StartM"][idx_ch]) 
        df.loc[idx_ch, "End"] = f(df["EndM"][idx_ch])
    df["length"] = df["End"] - df["Start"]
    return df



def create_ind_ibd_df(ibd_data = "/n/groups/reich/hringbauer/git/yamnaya/output/ibd/v43/ch_all.tsv",
                      min_cms = [8, 12, 16, 20], snp_cm = 220, 
                      min_cm = 6, sort_col = -1,
                      savepath="", output=True):
    """Create dataframe with summary statistics for each individual.
    Return this novel dataframe in hapROH format [IBD in cM]
    ibd_data: If string, what ibd file to load. Or IBD dataframe.
    savepath: If given: Save post-processed IBD dataframe to there.
    min_cms: What IBD lengths to use as cutoff in analysis [cM].
    snp_cm: Minimum Density of SNP per cM of IBD block.
    sort_col: Which min_cms col to use for sort. If <0 no sort conducted."""
    if isinstance(ibd_data, str): 
        df_ibd = pd.read_csv(ibd_data, sep="\t")
    else:
        df_ibd = ibd_data
        
    ### Filter. Be aggressive here for good set for  relatedness. Cutoff the slack
    df_ibd = filter_ibd_df(df_ibd, min_cm=min_cm, snp_cm=snp_cm, output=output) 
    
    ### Fish out the no IBD data
    if len(df_ibd)==0:
        df_res = pd.DataFrame(columns=['iid1','iid2', "max_IBD"])
        for m_cm in min_cms:
            df_res[f"sum_IBD>{m_cm}"] = 0
            df_res[f"n_IBD>{m_cm}"] = 0
    
    else: # In case there are IBD
        if output:
            print(df_ibd["ch"].value_counts())

        #df_ibd = df_ibd.sort_values(by = ["iid1", "iid2", "ch"]) # Sort so it is easier to split up
        grouped = df_ibd.groupby(['iid1','iid2'])

        df_res = pd.DataFrame(grouped.groups.keys())

        df_res.columns = ["iid1", "iid2"]

        ### Create the actual statistics
        stats = [roh_statistics_df(df, min_cms=min_cms) for _, df in grouped]
        stats = np.array(stats)

        df_res["max_IBD"] = stats[:, 0, 2] * 100

        ### Add Values for varying cM cutoffs:
        for j, m_cm in enumerate(min_cms):
            df_res[f"sum_IBD>{m_cm}"] = stats[:,j, 0] * 100
            df_res[f"n_IBD>{m_cm}"] = stats[:,j,1]

        if sort_col>=0:
            df_res = df_res.sort_values(by=f"sum_IBD>{min_cms[sort_col]}", ascending=False)  # Sort output by minimal Cutoff  
    
    ### Save if needed
    if len(savepath) > 0:
        df_res.to_csv(savepath, sep="\t", index=False)
        print(f"Saved {len(df_res)} individual IBD pairs to: {savepath}")

    return df_res

def to_hapsburg_ibd_df(path_segments = "../ped-sim/output/test.seg",
                   savepath = "", n=500, merge=False,
                   h5_path = "",
                   min_cm=[8, 12, 16, 20], snp_cm=220,
                   gap=0.5, min_len1=2, min_len2=4,
                   output=False, sort=True):
    """Load pd_sim output and post_process into Hapsburg
    Summary output. Return this dataframe.
    If savepath is given, save to there (tab-seperated)"""
    df1 = load_segment_file(path_segments)  # Load the full segment file, transfomred
    
    if merge:
        df1 = merge_called_blocks(df1, output=True)
        
    ### Pre-Process if h5 given
    if len(h5_path)>0:
        df1 = cap_ibd_boarders(df1, h5_path=h5_path)
        df1 = transform_to_snp_pos(df1, h5_path=h5_path)
        
    df_ibd = create_ind_ibd_df(ibd_data=df1,
                  min_cms=min_cm, snp_cm=snp_cm, min_cm=4,
                  sort_col=-1, savepath=savepath,
                  output=False)
    
    #assert(len(df_ibd)==n) # Sanity Check    
    return df_ibd
