#from flask import request
import os, re
import pwd
import datetime
from tempfile import mkstemp

util_dir = os.path.dirname(os.path.abspath(__file__))
#print(util_dir)
util_dir = re.sub('/$','',util_dir)

#host = "http://finsurf.biologie.ens.fr/"
# sample_dir = util_dir + "/static/data/samples/"
# gene_file = util_dir + "/static/data/2020-05-11_table_genes_FINSURF_regions.tsv"
# score_file = util_dir + "/static/data/scores_all_chroms_1e-4.tsv.gz"
# regulatory_file = util_dir + "/static/data/FINSURF_REGULATORY_REGIONS_GENES.bed.gz"

# ## for plotting
# rename_cols_table = util_dir + "/static/data/FINSURF_model_objects/rename_columns_model.tsv"
# numFeat = util_dir + "/static/data/NUM_FEATURES.tsv.gz"
# scaled_numFeat = util_dir + "/static/data/SCALED_NUM_FEATURES.tsv.gz"
# featCont_transition = util_dir + "/static/data/FULL_FC_transition.tsv.gz"
# featCont_transversion = util_dir + "/static/data/FULL_FC_transversion.tsv.gz"

# ## file size limit
# max_upload_allowed = 100 * 1024 * 1024 # maximum upload file size allowed by the server
# max_variant_file_size = 50*1024*1024 # if the variant file's size exceeds this number, propose to send email
# max_variants = 300 # if the number of variants entered in the textbox exceeds this number, propose to send email. The value 300 is for testing only. It can be 100000, depending on the CPU's force and disk reading speed
# max_gene_file_size = 10*1024*1024 # limit the size of the gene file uploaded by user (may be unnecessary?)

# max_lines = 20 ## maximum number of lines to be displayed

header = ["chrom", "pos", "end", "score", "id","ref","alt","vartype","vartrans","ucsc_link","el_id","genes"] # header for result file
table_header = """
	<thead>
    			<tr>
      				<th class="th-sm">Chrom</th>
      				<th class="th-sm">Pos</th>
      				<th class="th-sm">End</th>
                    <th class="th-sm">Score</th>
      				<th class="th-sm">ID</th>
      				<th class="th-sm">Ref</th>
      				<th class="th-sm">ALT</th>
      				<th class="th-sm">vartype</th>
      				<th class="th-sm">vartrans</th>
                    <th class="th-sm">ucsc</th>
                    <th class="th-sm">el_id</th>
      				<th class="th-sm">Gene</th>
    			</tr>
  		</thead>
"""

def make_tmp_file(method_name, out_format, dir=''):
    """Create a temporary file on the server.
    
    :param method_name: name of the tool (service) that uses the temporary file, used as prefix for the file's name
    :param out_format: format of the temporary file
    :dir: directory in which the temporary file will be created. If not specified, a temporary directory will be created. See @make_tmp_dir.
    
    :return: file name.

    """
    if dir == '':
        user = pwd.getpwuid(os.getuid())[0]
        tmp_dir = util_dir + "/../res/"
        dt = datetime.datetime.now()
        #tmp_dir += '/' + dt.strftime('%Y') + '/' + dt.strftime('%m') + '/' + dt.strftime('%d') + '/'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            os.chmod(tmp_dir,0o755)
    else:
        tmp_dir = dir
    dt = datetime.datetime.now()
    pref = method_name + '_' + dt.strftime('%Y') + '-' + dt.strftime('%m') + '-' + dt.strftime('%d') + '.' + dt.strftime('%H')+dt.strftime('%M')+dt.strftime('%S') + '_'

    if out_format != '':
        out_format = '.' + out_format
    fd, temp_path = mkstemp(out_format, pref, tmp_dir)
    os.chmod(temp_path, 0o755)
    return temp_path

# def save_gene_file():
# 	"""Save the genes entered or uploaded by user to a local file
# 	   :return: path to saved file
# 	"""
# 	tmp_gene_file_name = make_tmp_file('gene','txt','')
# 	if "gene" in request.form and request.form.get("gene") != '':
# 		genes = request.form.get("gene").strip().split("\n")
# 		f = open(tmp_gene_file_name, "w")
# 		for g in genes:
# 			f.write(g.strip() + "\n")
# 		f.close()
# 	elif request.files and "gene_file" in request.files:
# 		gene_file = request.files["gene_file"]
# 		gene_file.save(tmp_gene_file_name)	
			
# 	return tmp_gene_file_name

# def check_genes(file_name):
# 	"""Check if genes entered or uploaded by user are in the FINSURF database.
# 	:file_name: path to file containing genes entered by user
# 	:return: asso_genes: list of genes that have associated regulatory regions in FINSURF database
# 		 no_asso_genes: list of genes that don't have associated regulatory regions in FINSURF database
# 		 undef_genes: list of unrecognized genes
# 	"""
# 	#get genes that have associated regulatory regions
# 	asso_genes = os.popen("awk '(NR==FNR && $2==1) {a[$1]++; next} $1 in a' " + gene_file + " " + file_name).read()
# 	#get genes that don't have associated regulatory regions
# 	no_asso_genes = os.popen("awk '(NR==FNR && $2==0) {a[$1]++; next} $1 in a' " + gene_file + " " + file_name).read()
# 	#get genes unidentified
# 	undef_genes = os.popen("awk 'NR==FNR {a[$1]++; next} !($1 in a)' " + gene_file + " " + file_name).read()
# 	if asso_genes != '':
# 		asso_genes = asso_genes.strip().split("\n")
# 	else:
# 		asso_genes = []
# 	if no_asso_genes != '':
# 		no_asso_genes = no_asso_genes.strip().split("\n")
# 	else:
# 		no_asso_genes = []
# 	if undef_genes != '':
# 		undef_genes = undef_genes.strip().split("\n")
# 	else:
# 		undef_genes = []
# 	return (asso_genes, no_asso_genes, undef_genes)

# def print_genecheck(file_name):
# 	"""Build the html string to print the genes entered or uploaded by user according to their categories.
# 	  	In green color: genes that have associated regulatory regions in FINSURF database
# 		In orange color: gene that don't have associated regulatory regions in FINSURF database
# 		In red color: unrecognized genes
# 	   :return: html string  
# 	"""
# 	(asso_genes, no_asso_genes, undef_genes) = check_genes(file_name)
# 	gene_check_html = ''
# 	if len(asso_genes) > 0:
# 		gene_check_html += "<span style='color:green'>" + "<br/>".join(asso_genes) + "</span><br/>"
# 	if len(no_asso_genes) > 0:
# 		gene_check_html += "<span style='color:orange'>" + "<br/>".join(no_asso_genes) + "</span><br/>"
# 	if len(undef_genes) > 0:
# 		gene_check_html += "<span style='color:red'>" + "<br/>".join(undef_genes) + "</span>"
# 	if gene_check_html != '':
# 		gene_check_html = "<div  style='border:1px solid #ccc; border-radius:5px; overflow-y: scroll; max-height:100px; width:100%; font-size:14px;'>" + gene_check_html + "</div>"
# 	return gene_check_html


# def check_vcf_file():
# 	"""Check the variant file
# 	   :return: too_big: true if file is too big (> 50M)
# 		tmp_file_name: path to variant input file saved on local machine 
# 	"""
# 	variants = ''
# 	tmp_file_name = ''
# 	if request.files and 'variant_file' in request.files and bool(re.search(".gz$", request.files["variant_file"].filename)):
# 		tmp_file_name = make_tmp_file('variant','gz','')
# 	else:
# 		tmp_file_name = make_tmp_file('variant','vcf','')
	
# 	too_big = False
# 	if 'variant' in request.form and request.form.get('variant') != '':
# 		variants = request.form.get('variant').strip()
# 		f = open(tmp_file_name, "w")
# 		f.write(variants)
# 		f.close()
# 		if len(variants.split('\n')) > max_variants:
# 			too_big = True
# 	elif request.files and 'variant_file' in request.files:
# 		request.files['variant_file'].save(tmp_file_name)
# 		if os.stat(tmp_file_name).st_size > max_variant_file_size:
# 			too_big = True
	
# 	return (too_big, tmp_file_name)
				

# def make_link(filename):
# 	"""Convert local file path to url link
# 	"""
# 	return re.sub(util_dir,host,filename)

# def link_to_local(link):
# 	"""Convert url link to local file path
# 	"""
# 	return re.sub(host, util_dir, link)	

# def filtergene(gene_file, result_file):
# 	"""Filter the result by genes entered or uploaded by user
# 	   :return: filtered_file: path to the file containing the result filtered by genes
# 		    filtered_list: the list of result filtered by genes
# 	"""
# 	filtered_file = ''
# 	filtered_list = []
# 	filtered = os.popen("awk 'NF>0 && FNR==NR{a[$1]; next} {for (i in a) if (index($12, i)) print}' " + gene_file + " " + result_file + " | sort -ur -k4").read()
# 	if filtered != '':
# 		filtered_file = make_tmp_file('filtered_result','txt','')
# 		ff = open(filtered_file, "w")
# 		ff.write("#" + "\t".join(header) + "\n")
# 		ff.write(filtered.strip())
# 		ff.close()
# 		filtered = filtered.strip().split("\n")
# 		for i in filtered:
# 			filtered_list.append(i.split("\t"))
	
# 	return (filtered_file, filtered_list)

def get_size(fobj):
    if fobj.content_length:
        return fobj.content_length

    try:
        pos = fobj.tell()
        fobj.seek(0, os.SEEK_END)  #seek to end
        size = fobj.tell()
        fobj.seek(pos)  # back to original position
        return size
    except (AttributeError, IOError):
        pass

    # in-memory file object that doesn't support seeking or tell
    return 0  #assume small enough
