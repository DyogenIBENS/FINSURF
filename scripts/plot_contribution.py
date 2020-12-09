import argparse
import os
import sys

import pandas as pd

import tabix

import plotly.express as px
import plotly
import utils

## text size
SMALL_SIZE=12
MEDIUM_SIZE=20
BIGGER_SIZE=22

###############################################################################
# LOGGER

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s'
                    )
LOGGER = logging.getLogger(__name__)

###############################################################################
# DEFINITIONS

        
# Argument parser
# ===============

def argparser():
    parser = argparse.ArgumentParser(epilog=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--variant',
                        help='Variant position (format: "chrom:pos")',
                        type=str,
                        required=True
                        )

    parser.add_argument('--vartype',
                        help='Variant type',
                        type=str,
                        choices=['transition','transversion','not_SVN'],
                        required=True
                       )

    parser.add_argument('--rename_cols_table',
                        help='Table of model columns (old) with associated names for renaming (new).',
                        type=str,
                        required=True
                       )
    
    parser.add_argument('--numFeat_path',
                        help='Path to tabix table of numeric features.',
                        type=str,
                        required=True
                       )
                       
    parser.add_argument('--scaled_numFeat_path',
                        help='Path to tabix table of *scaled* numeric features.',
                        type=str,
                        required=True
                       )

    parser.add_argument('--featCont_transition_path',
                        help='Path to tabix table of tabix table of FC for transitions',
                        type=str,
                        required=True
                       )

    parser.add_argument('--featCont_transversion_path',
                        help='Path to tabix table of tabix table of FC for transversions',
                        type=str,
                        required=True
                       )

    return parser

# Main and arguments
# ------------------

def do_plot(variant_query, vartype, rename_cols_table, numFeat_path,  scaled_numFeat_path, featCont_transition_path, featCont_transversion_path):
    try:
        pos = variant_query.split(":")[1]
        variant_query += "-"
        variant_query += pos
    except Exception as e:
        LOGGER.exception("Error while trying to read the input variant string" + e )
        return ("Error while trying to read the input variant string",  None,  None,  None)

    renaming_table = pd.read_table(rename_cols_table,
                                   header=0,
                                   index_col=None,
                                   sep="\t"
                                  )                 
    map_rename = renaming_table.set_index("old")["new"]

    # Get the scaled numeric features
    try:
        bed_tabix = tabix.open(scaled_numFeat_path)
        query_res = bed_tabix.querys(variant_query)
        res = list(query_res)
        if not len(res)==1:
            raise RuntimeError("No record found or More than 1 hit.")
            return ("Error: More than 1 hit.",  None,  None,  None)

        scaled_features = pd.Series(res[0][4:],
                                    name='scaledNum',
                                    index=map_rename
                                    ).astype(float)
        
        # Here : we need to update the first feature value, corresponding to
        # the variant type ordinal value. 0 (default) for transition, 0.5 for
        # transversion, and 1 for indels.
        if vartype == "transversion":
            scaled_features.loc["Variant type"] = 0.5

        elif vartype == "indel":
            scaled_features.loc["Variant type"] = 1

        else:
            scaled_features.loc["Variant type"] = 0
        
    except Exception as e:
        LOGGER.exception("Error while Getting the numeric features")
        return ("Error while Getting the numeric features",  None,  None,  None)


    # Get the FC values
    try:
        if vartype=='transition':
            bed_tabix = tabix.open(featCont_transition_path)

        elif vartype=='transversion' or vartype == 'not_SNV':
            bed_tabix = tabix.open(featCont_transversion_path)

        elif vartype=='indel':
            #TODO : UPDATE WITH THE REAL TABLE ONCE GENERATED.
            # The values should match the transversions' one though.
            bed_tabix = tabix.open(featCont_transversion_path)

        query_res = bed_tabix.querys(variant_query)
        res = list(query_res)
        if not len(res)==1:
            raise RuntimeError("No record found or More than 1 hit.")
            return ("Error: No record found or More than 1 hit.",  None,  None,  None)

        feature_contributions = pd.Series(res[0][4:],
                                          name='featCont',
                                          index=map_rename
                                          ).astype(float)
            

    except Exception as e:
        LOGGER.exception("Error while getting the feature contributions")
        return ("Error while getting the feature contributions.",  None,  None,  None)

    # Tabix block to add for retrieving the "numeric features"
    try:
        bed_tabix = tabix.open(numFeat_path)
        query_res = bed_tabix.querys(variant_query)
        res = list(query_res)
        if not len(res)==1:
            raise RuntimeError("No record found or More than 1 hit.")
            return ("Error: No record found or More than 1 hit.",  None,  None,  None)

        raw_features = pd.Series(res[0][4:],
                                    name='rawNum',
                                    index=map_rename
                                    ).astype(float)

        if vartype == "transversion":
            raw_features.loc["Variant type"] = 1

        elif vartype == "indel":
            raw_features.loc["Variant type"] = 2

        else:
            raw_features.loc["Variant type"] = 0 
            
    except Exception as e:
        LOGGER.exception("Error while getting the numerice features")
        return ("Error while getting the numeric features.",  None,  None,  None)
    
    # Combine feature cotributions and scaled features
    df =  pd.concat([raw_features, scaled_features,feature_contributions], axis=1) 
    # Plot the bar chart
    fig = px.bar(df, x=df.index, y='scaledNum',color='featCont',
                        color_continuous_scale=[[0,'#145787'],[0.5,'white'],[1,'red']], 
                        range_color=[-0.06, 0.06])
    fig.update_traces(marker_line_color='#888888', marker_line_width=1, 
                        hovertemplate="<br>%{hovertext}",
                        hovertext = [('Raw: {:.4f}<br>Contribution: {:.4f}<br>'
                        ).format(row['rawNum'],
                                 row['featCont'])
                        for i, row in df.iterrows()],
                    hoverinfo = 'none',
                    hoverlabel={"namelength" :-1}, 
                )
    fig['layout']['yaxis1'].update(title='Scaled feature', range=[-1.1,1.1], autorange=False, titlefont=dict(size=MEDIUM_SIZE), tickfont=dict(size=SMALL_SIZE))
    fig['layout']['xaxis1'].update(title='', tickfont=dict(size=SMALL_SIZE), tickangle=-45)
    fig.update_layout(title= "Variant " + variant_query.split("-")[0],  hovermode='x unified',  coloraxis_colorbar=dict(title="Feature Contribution", titlefont=dict(size=MEDIUM_SIZE), titleside='right', tickfont=dict(size=SMALL_SIZE), thicknessmode="pixels", thickness=15))
    
    #write to files
    svg_file = utils.make_tmp_file('plot','svg','')	
    jpeg_file = utils.make_tmp_file('plot','jpeg','')
    png_file = utils.make_tmp_file('plot','png','')
    html_file = utils.make_tmp_file('plot','html','')
    plotly.offline.plot(fig, filename=svg_file+".html", image_filename=svg_file,  image='svg', auto_open=False)
    plotly.offline.plot(fig, filename=jpeg_file+".html", image_filename=jpeg_file, image='jpeg', auto_open=False)
    plotly.offline.plot(fig, filename=png_file+".html", image_filename=png_file, image='png', auto_open=False)
    fig.write_html(html_file)
    os.remove(svg_file)
    os.remove(jpeg_file)
    os.remove(png_file)
    LOGGER.info("Plotting the figure.")
    return (svg_file, jpeg_file, png_file, html_file)

###############################################################################
# MAIN

if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    
    (svg_file,  jpeg_file,  png_file,  html_file) = do_plot(args.variant, args.vartype, args.rename_cols_table, args.numFeat_path,  args.scaled_numFeat_path, args.featCont_transition_path, args.featCont_transversion_path)
    if jpeg_file == None:
        print(svg_file)
        LOGGER.info("Script ended with errors: " + svg_file )
    else:
        print("SVG: " + svg_file + "\nJPEG: " + jpeg_file + "\nPNG: " + png_file + "\nHTML: " + html_file)
        LOGGER.info("Script ended successfully.")
    sys.exit(0)
