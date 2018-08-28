# Read sample text file and generate binary feature of samples.

import process
import sys
if __name__ == "__main__":

  dict_file = sys.argv[1]
  sample_file = sys.argv[2]
  output_file = sys.argv[3]
  sample_dict = process.open_dict(dict_file)
  print("Feature size:", len(sample_dict))
  process.binarize_samples_files_part(sample_file, sample_dict, 1000, output_file)
  #process.save_sample(label, features, output_file)
