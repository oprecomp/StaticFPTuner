import random

array_name = "data"
n_datasets = 30
n_elements = 4000

print("#ifndef DATASET")
print("#define DATASET 0")
print("#endif")

for i in range(n_datasets):
    print("#if DATASET == %d" % (i))
    # Generate a dataset
    print("double h_StockPrice[]={")
    for i in range(n_elements):
        print(random.uniform(5.0, 30.0), end=",\n")
    print("};")
    print("#endif")

for i in range(n_datasets):
    print("#if DATASET == %d" % (i))
    # Generate a dataset
    print("double h_OptionStrike[]={")
    for i in range(n_elements):
        print(random.uniform(1.0, 100.0), end=",\n")
    print("};")
    print("#endif")


for i in range(n_datasets):
    print("#if DATASET == %d" % (i))
    # Generate a dataset
    print("double h_OptionYears[]={")
    for i in range(n_elements):
        print(random.uniform(0.25, 10.0), end=",\n")
    print("};")
    print("#endif")
