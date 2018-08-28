import process

# Check data having duplicate features in one sample
def check_data(fname, dictfile):
  wdict = process.open_dict(dictfile)
  lines = [line.rstrip('\n') for line in open(fname)]  
  #sample_num = len(lines)
  features = list()
  labels = list()
  feature_only = False
  len_max = 0
  len_max_dict = 0
  for line in lines: #[0:10]:
    label_features = line.split('\t')
    #print("label", label_features[0])
    #print("features", label_features[1])
    ftext = label_features[1]
    eles = ftext.split(',')
    #print("elements", eles)
    unique_features = []
    dict_features = []
    for e in eles:
      if e in unique_features :
        print("Having duplicate features", eles)
      else:
        unique_features.append(e)
      if e in wdict:
        dict_features.append(e)
    #print(unique_features)
    len_max = max(len_max, len(eles))
    len_max_dict = max(len_max_dict, len(dict_features))
  
  print("Length max:",len_max)
  print("Length max dict:",len_max_dict)
    

if __name__ == "__main__":
  check_data("../data/training-data-small.txt","../model/dict-small.npy")
  #check_data("../data/training-data-large.txt","../model/dict-large.npy")