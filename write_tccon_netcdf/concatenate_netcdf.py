from __future__ import print_function

import subprocess

"""
Code to concatenate netcdf files along the 'time' dimension using NCO's ncrcat
"""

def parse_file(file):
    """
    function to parse 
    """

    with open(file) as f:
        file_list = f.read().splitlines()

    return file_list

def parse_list(input):

    return input.split(',')

def execute(cmd):
    '''
    function to execute a unix command and print the output as it is produced
    '''
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def main():
    parser = argparse.ArgumentParser(description="Code to concatenate netcdf files along the 'time' dimension using NCO's ncrcat")
    parser.add_argument('-p',--'path',help='full path to the folder where the concatenated netcdf file will be saved')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f','--file',type=parse_file,help='full path to file containing a list of netcdf files to concatenate, with one file path per line')
    group.add_argument('-l','--list',type=parse_list,help='comma separated list of netcdf files to concatenate; make sure they are in order !')
    
    args = parser.parse_args()

    if args.file:
        file_list = args.file
    elif args.list:
        file_list = args.list

    nsites = len(set([f[:2] for f in file_list]))
    if nsites != 1:
        sys.exit('Can only concatenate files from the same site')

    site_abbrv = file_list[0][:2]
    start_date = file_list[0][2:10]
    end_date = file_list[-1][11:19]

    output_file = "{}{}_{}.public.nc".format(site_abbrv,start_date,end_date)

    if args.path:
        output_file = os.path.join(args.path,output_file)

    for line in execute(['ncrcat']+file_list+[output_file]):
        print(line, end="")


if __name__=="__main__": # execute only when the code is run by itself, and not when it is imported
    main()


