import h5py

with h5py.File("data/temp/vimeo_test_sample.hdf5", 'r') as hf:
    print(hf)
    print(hf.keys())
    print(hf["data_lr"].shape)
    print(hf["data_hr"].shape)

with h5py.File("data/temp/vimeo_subset_v6.h5", 'r') as hf:
    print(hf)
    print(hf.keys())
    print(hf["data_lr"].shape)
    print(hf["data_hr"].shape)

print("poooop")
