import sys
import csv
import subprocess

# Pulp with SmallFloat extensions
def set_coefficient_bits(prec):
    if(prec <= 3):                   # float8
        return 5
    elif(prec > 3 and prec <= 8):    # float16ext
        return 8
    elif(prec > 8 and prec <= 11):   # float16
        return 5
    elif(prec > 11 and prec <= 24):  # float32
        return 8
    elif(prec > 24 and prec <= 53):  # float64
        return 11
    else:
        raise Exception

def init_params(config_vals):
    result = []

    result.append(" -DFRAC_K=%d" % (config_vals[0] - 1))
    result.append(" -DFRAC_CND=%d" % (config_vals[1] - 1))
    result.append(" -DFRAC_D=%d" % (config_vals[2] - 1))
    result.append(" -DFRAC_SQRTT=%d" % (config_vals[3] - 1))
    result.append(" -DFRAC_EXPRT=%d" % (config_vals[4] - 1))
    result.append(" -DFRAC_SX=%d" % (config_vals[5] - 1))
    result.append(" -DFRAC_RV=%d" % (config_vals[6] - 1))
    result.append(" -DFRAC_T=%d" % (config_vals[7] - 1))
    result.append(" -DFRAC_CALLRESULT=%d" % (config_vals[8] - 1))
    result.append(" -DFRAC_PUTRESULT=%d" % (config_vals[9] - 1))
    result.append(" -DFRAC_TEMP1=%d" % (config_vals[10] - 1))
    result.append(" -DFRAC_TEMP2=%d" % (config_vals[11] - 1))
    result.append(" -DFRAC_TEMP3=%d" % (config_vals[12] - 1))
    result.append(" -DFRAC_TEMP4=%d" % (config_vals[13] - 1))
    result.append(" -DFRAC_TEMP5=%d" % (config_vals[14] - 1))



    result.append(" -DEXP_K=%d" % set_coefficient_bits(config_vals[0]))
    result.append(" -DEXP_CND=%d" % set_coefficient_bits(config_vals[1]))
    result.append(" -DEXP_D=%d" % set_coefficient_bits(config_vals[2]))
    result.append(" -DEXP_SQRTT=%d" % set_coefficient_bits(config_vals[3]))
    result.append(" -DEXP_EXPRT=%d" % set_coefficient_bits(config_vals[4]))
    result.append(" -DEXP_SX=%d" % set_coefficient_bits(config_vals[5]))
    result.append(" -DEXP_RV=%d" % set_coefficient_bits(config_vals[6]))
    result.append(" -DEXP_T=%d" % set_coefficient_bits(config_vals[7]))
    result.append(" -DEXP_CALLRESULT=%d" % set_coefficient_bits(config_vals[8]))
    result.append(" -DEXP_PUTRESULT=%d" % set_coefficient_bits(config_vals[9]))
    result.append(" -DEXP_TEMP1=%d" % set_coefficient_bits(config_vals[10]))
    result.append(" -DEXP_TEMP2=%d" % set_coefficient_bits(config_vals[11]))
    result.append(" -DEXP_TEMP3=%d" % set_coefficient_bits(config_vals[12]))
    result.append(" -DEXP_TEMP4=%d" % set_coefficient_bits(config_vals[13]))
    result.append(" -DEXP_TEMP5=%d" % set_coefficient_bits(config_vals[14]))


    return "".join(result)

with open(sys.argv[1], 'r') as config_file:
    reader = csv.reader(config_file)
    row = next(reader)
    if row[-1] == '':
        del row[-1]
    config_vals = [int(x) for x in row]
    ext_cflags = init_params(config_vals)
    make_process = subprocess.Popen(
            "make clean all -f Makefile_flex CONF_MODE=file EXT_CFLAGS=\""
            + ext_cflags + "\" OUTPUT_DIR=\"" + sys.argv[2] + "\" DATASET=" + 
            sys.argv[3], shell=True, stderr=subprocess.STDOUT)
    make_process.wait()
